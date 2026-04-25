import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel


ROOT_DIR = Path(__file__).resolve().parent
PYTHON_PACKAGE_DIR = ROOT_DIR / "python-package"
if str(PYTHON_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE_DIR))

from insightface.app import FaceAnalysis  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CACHE_FILENAME = ".face_embeddings_cache.pkl"


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
    processing_ms: float
    image_width: int
    image_height: int
    faces: List[FaceResult]


class PeopleResponse(BaseModel):
    people: List[str]
    image_count: Dict[str, int]
    threshold: float


class RuntimeResponse(BaseModel):
    onnxruntime_version: str
    available_providers: List[str]
    active_providers: List[str]
    det_size: int
    device: str


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
        self.det_size = det_size
        self.providers = providers
        self.ctx_id = ctx_id
        self.cache_path = face_db / CACHE_FILENAME
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self.people: Dict[str, np.ndarray] = {}
        self.image_count: Dict[str, int] = {}

    def load(self) -> None:
        if not self.face_db.exists():
            self.face_db.mkdir(parents=True, exist_ok=True)

        cache = self._load_cache()
        next_cache = {"images": {}}
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
                cache_key = self._cache_key(image_path)
                image_stat = image_path.stat()
                cached = cache.get("images", {}).get(cache_key)
                if (
                    cached
                    and cached.get("mtime_ns") == image_stat.st_mtime_ns
                    and cached.get("size") == image_stat.st_size
                    and cached.get("person") == person_dir.name
                ):
                    embeddings.append(np.asarray(cached["embedding"], dtype=np.float32))
                    next_cache["images"][cache_key] = cached
                    continue

                img = cv2.imread(str(image_path))
                if img is None:
                    print(f"skip unreadable image: {image_path}")
                    continue

                faces = self.app.get(img)
                if not faces:
                    print(f"skip no-face image: {image_path}")
                    continue

                face = self._largest_face(faces)
                embedding = face.normed_embedding.astype(np.float32)
                embeddings.append(embedding)
                next_cache["images"][cache_key] = {
                    "person": person_dir.name,
                    "mtime_ns": image_stat.st_mtime_ns,
                    "size": image_stat.st_size,
                    "embedding": embedding,
                }

            if embeddings:
                mean_embedding = np.mean(np.asarray(embeddings), axis=0)
                mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
                people[person_dir.name] = mean_embedding.astype(np.float32)
                image_count[person_dir.name] = len(embeddings)
            else:
                image_count[person_dir.name] = 0

        self.people = people
        self.image_count = image_count
        self._save_cache(next_cache)
        print(f"loaded {len(self.people)} people from {self.face_db}")

    def identify_image_bytes(self, image_bytes: bytes) -> IdentifyResponse:
        img = self._decode_image(image_bytes)

        started_at = time.perf_counter()
        faces = self.app.get(img)
        processing_ms = (time.perf_counter() - started_at) * 1000
        height, width = img.shape[:2]
        return self.identify_faces(faces, processing_ms, width, height)

    def identify_faces(
        self,
        faces,
        processing_ms: float = 0.0,
        image_width: int = 0,
        image_height: int = 0,
    ) -> IdentifyResponse:
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
            processing_ms=round(processing_ms, 1),
            image_width=image_width,
            image_height=image_height,
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

    def _cache_key(self, image_path: Path) -> str:
        return image_path.relative_to(self.face_db).as_posix()

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {"images": {}}
        try:
            with self.cache_path.open("rb") as file:
                cache = pickle.load(file)
            if isinstance(cache, dict) and isinstance(cache.get("images"), dict):
                return cache
        except Exception as exc:
            print(f"ignore invalid cache {self.cache_path}: {exc}")
        return {"images": {}}

    def _save_cache(self, cache: dict) -> None:
        try:
            with self.cache_path.open("wb") as file:
                pickle.dump(cache, file)
        except Exception as exc:
            print(f"cannot save cache {self.cache_path}: {exc}")


def create_app(database: FaceDatabase) -> FastAPI:
    app = FastAPI(title="InsightFace Local Face API")

    @app.on_event("startup")
    def startup() -> None:
        database.load()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "people_count": len(database.people)}

    @app.get("/")
    def web_app() -> FileResponse:
        return FileResponse(ROOT_DIR / "web" / "index.html")

    @app.get("/people", response_model=PeopleResponse)
    def people() -> PeopleResponse:
        return PeopleResponse(
            people=sorted(database.people.keys()),
            image_count=database.image_count,
            threshold=database.threshold,
        )

    @app.get("/runtime", response_model=RuntimeResponse)
    def runtime() -> RuntimeResponse:
        return RuntimeResponse(
            onnxruntime_version=ort.__version__,
            available_providers=ort.get_available_providers(),
            active_providers=database.providers,
            det_size=database.det_size,
            device="gpu" if database.ctx_id >= 0 else "cpu",
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
        started_at = time.perf_counter()
        image_bytes = await file.read()
        try:
            result = database.identify_image_bytes(image_bytes)
            total_ms = (time.perf_counter() - started_at) * 1000
            matches = ", ".join(
                f"{face.match.name or 'Unknown'}:{face.match.score:.3f}"
                for face in result.faces
            )
            print(
                "[identify] "
                f"bytes={len(image_bytes)} "
                f"image={result.image_width}x{result.image_height} "
                f"infer={result.processing_ms:.1f}ms "
                f"total={total_ms:.1f}ms "
                f"faces={result.face_count} "
                f"matches=[{matches}]",
                flush=True,
            )
            return result
        except ValueError as exc:
            print(f"[identify:error] bytes={len(image_bytes)} detail={exc}", flush=True)
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
