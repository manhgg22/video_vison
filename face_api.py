import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parent
PYTHON_PACKAGE_DIR = ROOT_DIR / "python-package"
if str(PYTHON_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_DIR))

from insightface.app import FaceAnalysis  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MatchResult(BaseModel):
    name: Optional[str]
    score: float
    matched: bool


class FaceResult(BaseModel):
    bbox: List[int]
    det_score: float
    match: MatchResult


class IdentifyResponse(BaseModel):
    threshold: float
    face_count: int
    faces: List[FaceResult]


class PeopleResponse(BaseModel):
    people: List[str]
    image_count: Dict[str, int]
    threshold: float


class FaceDatabase:
    def __init__(
        self,
        face_db: Path,
        threshold: float,
        det_size: int,
        providers: List[str],
        ctx_id: int,
    ) -> None:
        self.face_db = face_db
        self.threshold = threshold
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self.people: Dict[str, np.ndarray] = {}
        self.image_count: Dict[str, int] = {}

    def load(self) -> None:
        if not self.face_db.exists():
            self.face_db.mkdir(parents=True, exist_ok=True)

        people: Dict[str, np.ndarray] = {}
        image_count: Dict[str, int] = {}

        for person_dir in sorted(p for p in self.face_db.iterdir() if p.is_dir()):
            embeddings = []
            image_paths = [
                p
                for p in person_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ]

            for image_path in image_paths:
                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"skip unreadable image: {image_path}")
                    continue

                faces = self.app.get(img)
                if not faces:
                    print(f"skip no-face image: {image_path}")
                    continue

                face = self._largest_face(faces)
                embeddings.append(face.normed_embedding)

            if embeddings:
                mean_embedding = np.mean(np.asarray(embeddings), axis=0)
                mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
                people[person_dir.name] = mean_embedding.astype(np.float32)
                image_count[person_dir.name] = len(embeddings)
            else:
                image_count[person_dir.name] = 0

        self.people = people
        self.image_count = image_count
        print(f"loaded {len(self.people)} people from {self.face_db}")

    def identify_image_bytes(self, image_bytes: bytes) -> IdentifyResponse:
        img = self._decode_image(image_bytes)

        faces = self.app.get(img)
        return self.identify_faces(faces)

    def identify_faces(self, faces) -> IdentifyResponse:
        results = []
        for face in faces:
            match = self.match_embedding(face.normed_embedding)
            bbox = [int(x) for x in face.bbox]
            results.append(
                FaceResult(
                    bbox=bbox,
                    det_score=float(face.det_score),
                    match=MatchResult(
                        name=match[0],
                        score=match[1],
                        matched=match[1] >= self.threshold,
                    ),
                )
            )

        return IdentifyResponse(
            threshold=self.threshold,
            face_count=len(results),
            faces=results,
        )

    def verify_image_bytes(self, person_name: str, image_bytes: bytes) -> MatchResult:
        if person_name not in self.people:
            raise KeyError(person_name)

        img = self._decode_image(image_bytes)
        faces = self.app.get(img)
        if not faces:
            return MatchResult(name=person_name, score=0.0, matched=False)

        person_embedding = self.people[person_name]
        best_for_person = max(
            float(np.dot(face.normed_embedding, person_embedding)) for face in faces
        )

        return MatchResult(
            name=person_name,
            score=best_for_person,
            matched=best_for_person >= self.threshold,
        )

    def match_embedding(self, embedding: np.ndarray) -> tuple[Optional[str], float]:
        if not self.people:
            return None, 0.0

        best_name = None
        best_score = -1.0
        for name, known_embedding in self.people.items():
            score = float(np.dot(embedding, known_embedding))
            if score > best_score:
                best_name = name
                best_score = score

        return best_name, best_score

    @staticmethod
    def _largest_face(faces):
        def area(face) -> float:
            x1, y1, x2, y2 = face.bbox
            return float(max(0, x2 - x1) * max(0, y2 - y1))

        return max(faces, key=area)

    @staticmethod
    def _decode_image(image_bytes: bytes):
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Cannot decode image")
        return img


def create_app(database: FaceDatabase) -> FastAPI:
    app = FastAPI(title="InsightFace Local Face API")

    @app.on_event("startup")
    def startup() -> None:
        database.load()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "people_count": len(database.people)}

    @app.get("/people", response_model=PeopleResponse)
    def people() -> PeopleResponse:
        return PeopleResponse(
            people=sorted(database.people.keys()),
            image_count=database.image_count,
            threshold=database.threshold,
        )

    @app.post("/reload-db", response_model=PeopleResponse)
    def reload_db() -> PeopleResponse:
        database.load()
        return PeopleResponse(
            people=sorted(database.people.keys()),
            image_count=database.image_count,
            threshold=database.threshold,
        )

    @app.post("/identify", response_model=IdentifyResponse)
    async def identify(file: UploadFile = File(...)) -> IdentifyResponse:
        image_bytes = await file.read()
        try:
            return database.identify_image_bytes(image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/verify/{person_name}", response_model=MatchResult)
    async def verify(person_name: str, file: UploadFile = File(...)) -> MatchResult:
        image_bytes = await file.read()
        try:
            return database.verify_image_bytes(person_name, image_bytes)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown person: {person_name}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-db", default="face_db")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("FACE_THRESHOLD", "0.45")))
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


args = parse_args()
providers = ["CPUExecutionProvider"]
ctx_id = -1
if args.gpu:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ctx_id = 0

database = FaceDatabase(
    face_db=(ROOT_DIR / args.face_db).resolve(),
    threshold=args.threshold,
    det_size=args.det_size,
    providers=providers,
    ctx_id=ctx_id,
)
app = create_app(database)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
