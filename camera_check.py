import argparse
import time

import cv2
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    return parser.parse_args()


def identify_frame(api_url: str, frame, jpeg_quality: int) -> dict:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not ok:
        return {"faces": []}

    files = {"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
    response = requests.post(f"{api_url.rstrip('/')}/identify", files=files, timeout=10)
    response.raise_for_status()
    return response.json()


def draw_results(frame, result: dict) -> None:
    for face in result.get("faces", []):
        x1, y1, x2, y2 = face["bbox"]
        match = face["match"]
        matched = match["matched"]
        name = match["name"] if matched else "Unknown"
        score = match["score"]
        color = (0, 200, 0) if matched else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    last_check = 0.0
    last_result = {"faces": []}
    last_error = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        if now - last_check >= args.interval:
            last_check = now
            try:
                last_result = identify_frame(args.api, frame, args.jpeg_quality)
                last_error = ""
            except requests.RequestException as exc:
                last_error = str(exc)

        draw_results(frame, last_result)
        if last_error:
            cv2.putText(
                frame,
                "API error",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("InsightFace camera check - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
