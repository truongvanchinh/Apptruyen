"""Web server: tải PDF/ảnh manga lên, trích lời thoại, xem/sửa, tải file .txt."""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import extractor

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_MB = 200
JOB_TTL_SECONDS = 6 * 3600

app = FastAPI(title="Manga Dialogue Extractor")

# OCR tốn CPU/RAM: xử lý lần lượt từng job
executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def job_dir(job_id: str) -> Path:
    return DATA_DIR / job_id


def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job")
    return job


def cleanup_old_jobs() -> None:
    now = time.time()
    with jobs_lock:
        for job_id in [j for j, v in jobs.items() if now - v["created"] > JOB_TTL_SECONDS]:
            jobs.pop(job_id, None)
            shutil.rmtree(job_dir(job_id), ignore_errors=True)


def run_job(job_id: str, path: Path, engine: str, fix: bool) -> None:
    job = jobs[job_id]
    try:
        total, results = extractor.process_document(path, engine=engine, fix=fix)
        job["total"] = total
        job["status"] = "running"
        job["message"] = f"Đang xử lý trang 1/{total}"
        for page, image in results:
            cv2.imwrite(
                str(job_dir(job_id) / f"page_{page.page}.jpg"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            job["pages"].append(page.to_dict())
            job["done"] = page.page
            if job.get("cancel"):
                job["status"] = "cancelled"
                job["message"] = f"Đã dừng ở trang {page.page}/{total}"
                return
            job["message"] = f"Đang xử lý trang {min(page.page + 1, total)}/{total}"
        job["status"] = "done"
        job["message"] = "Hoàn tất"
    except Exception as exc:  # báo lỗi lên giao diện
        job["status"] = "error"
        job["message"] = f"Lỗi: {exc}"


@app.get("/api/engines")
def engines():
    return {"engines": list(extractor.available_engines())}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), engine: str = Form(""), fix: bool = Form(True)):
    cleanup_old_jobs()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, "Chỉ hỗ trợ PDF, JPG, PNG, WEBP")

    available = extractor.available_engines()
    if not available:
        raise HTTPException(500, "Chưa cài EasyOCR (pip install easyocr)")
    if engine not in available:
        engine = available[0]

    job_id = uuid.uuid4().hex[:12]
    folder = job_dir(job_id)
    folder.mkdir(parents=True)
    path = folder / f"input{suffix}"

    size = 0
    with path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                out.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, f"File vượt quá {MAX_UPLOAD_MB}MB")
            out.write(chunk)

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "engine": engine,
            "fix": fix,
            "status": "queued",
            "message": "Đang chờ xử lý",
            "total": 0,
            "done": 0,
            "pages": [],
            "created": time.time(),
        }
    executor.submit(run_job, job_id, path, engine, fix)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, since: int = 0):
    """`since`: chỉ trả các trang mới (trang > since) để giao diện cập nhật dần."""
    job = get_job(job_id)
    data = {k: v for k, v in job.items() if k not in ("pages", "cancel")}
    data["pages"] = [p for p in job["pages"] if p["page"] > since]
    return data


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    get_job(job_id)["cancel"] = True
    return {"ok": True}


@app.get("/api/jobs/{job_id}/pages/{number}.jpg")
def page_image(job_id: str, number: int):
    get_job(job_id)
    path = job_dir(job_id) / f"page_{number}.jpg"
    if not path.exists():
        raise HTTPException(404, "Trang chưa sẵn sàng")
    return FileResponse(path, media_type="image/jpeg")


class Region(BaseModel):
    x: int
    y: int
    w: int
    h: int


@app.post("/api/jobs/{job_id}/pages/{number}/ocr")
def ocr_region(job_id: str, number: int, region: Region):
    """OCR lại một vùng người dùng tự khoanh trên trang (bong bóng bị bỏ sót)."""
    job = get_job(job_id)
    path = job_dir(job_id) / f"page_{number}.jpg"
    if not path.exists():
        raise HTTPException(404, "Trang chưa sẵn sàng")
    image = extractor.load_image(path)
    bubble = extractor.Bubble(**region.model_dump())
    return {"text": extractor.ocr_bubble(image, bubble, job["engine"], fix=job["fix"])}


@app.get("/api/jobs/{job_id}/dialogue.txt", response_class=PlainTextResponse)
def dialogue(job_id: str):
    job = get_job(job_id)
    pages = [extractor.PageResult(p["page"], p["width"], p["height"],
                                  [extractor.Bubble(**b) for b in p["bubbles"]])
             for p in job["pages"]]
    return extractor.dialogue_text(pages)


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
