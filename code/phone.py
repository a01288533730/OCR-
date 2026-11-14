"""
HRC Steel ID OCR - web demo cho điện thoại (test model fine-tune).

Chạy:
    python demo_phone.py              (HTTP)
    python demo_phone.py --https      (HTTPS, cần cert.pem + key.pem cạnh file này)

Trên điện thoại (cùng Wi-Fi với máy tính): mở http://<IP máy tính>:8000
- Nút "Chụp ảnh" mở camera gốc của điện thoại, dùng được cả với HTTP.
- Tab "Camera khung ngắm" (xem trước + khung cắt quanh ID) cần HTTPS hoặc localhost.
"""
import argparse
import io
import json
import logging
import re
import socket
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import paddle
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from paddleocr import TextDetection, TextRecognition
from PIL import Image, ImageOps

# =========================
# CẤU HÌNH
# =========================
BASE_DIR = Path(__file__).resolve().parent

REC_MODEL_NAME = "PP-OCRv6_medium_rec"
DET_MODEL_NAME = "PP-OCRv6_medium_det"

# Thư mục model fine-tune đã export (có inference.yml, inference.pdiparams, ...).
# Sửa đường dẫn này cho đúng máy của bạn.
FINETUNE_DIR = r"D:\HAO\PaddleOCR\inference\HRC_rec_epoch74"

MODEL_LABELS = {"finetune": "Fine-tune (epoch 74)", "base": "Model gốc"}
MODEL_ORDER = ("finetune", "base")

MAX_FILE_SIZE = 15 * 1024 * 1024   # giới hạn dung lượng ảnh tải lên
MAX_IMAGE_SIDE = 2000              # thu nhỏ ảnh quá lớn ở phía server

# Lưu ảnh + kết quả để đánh giá lại / bổ sung dữ liệu train sau này.
SAVE_UPLOADS = True
UPLOAD_DIR = BASE_DIR / "demo_uploads"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HRC Steel ID OCR - Phone Demo")

det_model = None
rec_models = {}
device_name = "cpu"

model_lock = threading.Lock()   # tránh nhiều request cùng dùng model/GPU
log_lock = threading.Lock()

ID_PATTERN = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")


# =========================
# KHỞI TẠO MODEL
# =========================
def warm_up_models():
    """Chạy thử mỗi model một lần để lần quét đầu không bị chậm và lỗi (nếu có) hiện ngay."""
    blank = np.full((320, 320, 3), 255, dtype=np.uint8)
    try:
        list(det_model.predict(input=blank))
    except Exception:
        logger.exception("Warm-up detection thất bại.")
    for key, model in rec_models.items():
        try:
            list(model.predict(input=blank[:48], batch_size=1))
        except Exception:
            logger.exception("Warm-up recognition (%s) thất bại.", key)
    logger.info("Warm-up model hoàn tất.")


@app.on_event("startup")
def load_models():
    global det_model, device_name

    use_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    device_name = "gpu:0" if use_gpu else "cpu"
    if not use_gpu:
        logger.warning("Không thấy GPU CUDA, chạy bằng CPU (sẽ chậm hơn).")
    paddle.set_device(device_name)
    logger.info("Thiết bị: %s", device_name)

    det_model = TextDetection(model_name=DET_MODEL_NAME, device=device_name)
    logger.info("Detection model đã khởi tạo.")

    if Path(FINETUNE_DIR).is_dir():
        rec_models["finetune"] = TextRecognition(
            model_name=REC_MODEL_NAME, model_dir=FINETUNE_DIR, device=device_name
        )
        logger.info("Recognition fine-tune đã khởi tạo từ: %s", FINETUNE_DIR)
    else:
        logger.warning(
            "Không thấy thư mục model fine-tune: %s (bỏ qua, chỉ dùng model gốc).", FINETUNE_DIR
        )

    try:
        rec_models["base"] = TextRecognition(model_name=REC_MODEL_NAME, device=device_name)
        logger.info("Recognition model gốc đã khởi tạo.")
    except Exception:
        logger.exception("Không tải được model gốc.")

    if not rec_models:
        raise RuntimeError("Không có model nhận diện nào khả dụng.")

    warm_up_models()


@app.get("/health")
def health():
    return {
        "status": "ok" if det_model is not None and rec_models else "loading",
        "device": device_name,
        "models": {key: key in rec_models for key in MODEL_ORDER},
        "labels": MODEL_LABELS,
    }


# =========================
# XỬ LÝ ẢNH
# =========================
def result_to_dict(item):
    """Chuẩn hóa kết quả JSON từ các phiên bản PaddleOCR."""
    data = getattr(item, "json", None)

    if callable(data):
        data = data()

    if isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        return {}

    data = data.get("res", data)
    return data if isinstance(data, dict) else {}


def recognize(model, image_bgr):
    """Nhận diện chữ trên một vùng ảnh (mảng BGR). Trả về (text, score)."""
    best_text, best_score = "", 0.0

    for item in model.predict(input=image_bgr, batch_size=1) or []:
        res = result_to_dict(item)
        text = str(res.get("rec_text", "")).strip()

        try:
            score = float(res.get("rec_score", 0.0))
        except (ValueError, TypeError):
            score = 0.0

        # Chỉ giữ kết quả có chữ/số phù hợp với ID.
        if text and re.search(r"[A-Za-z0-9]", text):
            if not best_text or score > best_score:
                best_text, best_score = text, score

    return best_text, best_score


def detect_boxes(image_bgr, width, height):
    """Dò vùng chữ, trả về danh sách [x1, y1, x2, y2]."""
    polygons = []
    for item in det_model.predict(input=image_bgr) or []:
        res = result_to_dict(item)
        polygons = res.get("dt_polys", []) or []
        if len(polygons):
            break

    boxes = []
    for polygon in polygons:
        points = np.asarray(polygon).reshape(-1, 2)
        if len(points) < 4:
            continue

        x1 = max(0, int(np.min(points[:, 0])))
        y1 = max(0, int(np.min(points[:, 1])))
        x2 = min(width, int(np.max(points[:, 0])))
        y2 = min(height, int(np.max(points[:, 1])))

        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])

    return boxes


def read_boxes(image_bgr, boxes, model_keys):
    """Nhận diện chữ trong từng vùng bằng từng model đã chọn."""
    results = {key: [] for key in model_keys}

    for x1, y1, x2, y2 in boxes:
        crop = np.ascontiguousarray(image_bgr[y1:y2, x1:x2])

        for key in model_keys:
            text, score = recognize(rec_models[key], crop)
            if text:
                results[key].append({
                    "text": text,
                    "score": round(score, 4),
                    "confidence_percent": round(score * 100, 2),
                    "bbox": [x1, y1, x2, y2],
                })

    return results


def process_image(image, model_keys, skip_det):
    started = time.perf_counter()

    image = image.convert("RGB")
    width, height = image.size
    # PaddleOCR coi mảng numpy là ảnh BGR (như cv2.imread), giống lúc train (img_mode: BGR).
    image_bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    full_box = [[0, 0, width, height]]
    fallback = False

    with model_lock:
        if skip_det:
            results = read_boxes(image_bgr, full_box, model_keys)
        else:
            boxes = detect_boxes(image_bgr, width, height)
            results = (
                read_boxes(image_bgr, boxes, model_keys)
                if boxes else {key: [] for key in model_keys}
            )

            # Ảnh chụp sát ID thường không dò được vùng chữ: đọc cả ảnh làm phương án dự phòng.
            if not any(results.values()):
                results = read_boxes(image_bgr, full_box, model_keys)
                fallback = True

    payload = {}
    for key in model_keys:
        detections = sorted(results[key], key=lambda d: d["score"], reverse=True)
        best = detections[0] if detections else None
        payload[key] = {
            "label": MODEL_LABELS[key],
            "text": best["text"] if best else "",
            "confidence_percent": best["confidence_percent"] if best else 0.0,
            "detections": detections,
        }

    return {
        "status": "success",
        "results": payload,
        "skip_det": skip_det,
        "fallback_full_image": fallback,
        "image_width": width,
        "image_height": height,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
    }


# =========================
# LƯU ẢNH + LOG
# =========================
def append_jsonl(path, record):
    with log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_upload(upload_id, image, result):
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        image.save(UPLOAD_DIR / f"{upload_id}.jpg", format="JPEG", quality=92)

        append_jsonl(UPLOAD_DIR / "log.jsonl", {
            "id": upload_id,
            "time": datetime.now().isoformat(timespec="seconds"),
            "image_size": [result["image_width"], result["image_height"]],
            "skip_det": result["skip_det"],
            "fallback_full_image": result["fallback_full_image"],
            "results": {
                key: {
                    "text": r["text"],
                    "confidence_percent": r["confidence_percent"],
                    "count": len(r["detections"]),
                }
                for key, r in result["results"].items()
            },
        })
    except Exception:
        logger.exception("Không lưu được ảnh/log.")


# =========================
# API
# =========================
@app.post("/ocr/predict")
async def predict(
    file: UploadFile = File(...),
    models: str = Form("both"),
    skip_det: bool = Form(False),
):
    if det_model is None or not rec_models:
        raise HTTPException(status_code=503, detail="Model OCR chưa được khởi tạo.")

    wanted = {
        "both": list(MODEL_ORDER),
        "finetune": ["finetune"],
        "base": ["base"],
    }.get(models)

    if wanted is None:
        raise HTTPException(status_code=400, detail="Giá trị 'models' không hợp lệ.")

    model_keys = [key for key in wanted if key in rec_models]
    if not model_keys:
        raise HTTPException(status_code=400, detail="Model đã chọn hiện chưa khả dụng trên server.")

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(status_code=400, detail="File ảnh đang trống.")

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Ảnh vượt quá giới hạn dung lượng.")

        try:
            with Image.open(io.BytesIO(contents)) as source:
                source.verify()

            with Image.open(io.BytesIO(contents)) as source:
                # Xoay theo EXIF để ảnh đúng chiều như trên điện thoại.
                image = ImageOps.exif_transpose(source).convert("RGB")

        except (OSError, ValueError):
            raise HTTPException(status_code=400, detail="File không phải ảnh hợp lệ hoặc ảnh bị hỏng.")

        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

        upload_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        result = await run_in_threadpool(process_image, image, model_keys, skip_det)
        result["id"] = upload_id

        if SAVE_UPLOADS:
            await run_in_threadpool(save_upload, upload_id, image, result)

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Lỗi xử lý ảnh OCR")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    finally:
        await file.close()


@app.post("/ocr/feedback")
async def feedback(
    upload_id: str = Form(..., alias="id"),
    correct_text: str = Form(...),
):
    """Ghi lại ID đúng do người dùng nhập, để tính độ chính xác sau này."""
    text = correct_text.strip()

    if not ID_PATTERN.match(upload_id):
        raise HTTPException(status_code=400, detail="Mã ảnh không hợp lệ.")
    if not text or len(text) > 100:
        raise HTTPException(status_code=400, detail="ID đúng không hợp lệ.")

    append_jsonl(UPLOAD_DIR / "feedback.jsonl", {
        "id": upload_id,
        "correct_text": text,
        "time": datetime.now().isoformat(timespec="seconds"),
    })
    return {"status": "ok"}


# =========================
# GIAO DIỆN WEB
# =========================
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>HRC Steel ID OCR</title>
<style>
:root {
    --bg: #0b1220; --panel: #111c30; --panel-2: #0a1425; --line: #28364c;
    --text: #e8eef8; --muted: #9eacc2; --blue: #2878f0; --btn: #26364e;
    --green: #46df9b; --amber: #ffb020; --red: #ff6b6b;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body {
    margin: 0; background: var(--bg); color: var(--text);
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    -webkit-text-size-adjust: 100%;
}
header { padding: 16px 16px 12px; background: var(--panel); border-bottom: 1px solid var(--line); }
header h1 { margin: 0 0 4px; font-size: 20px; }
header p { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.5; }
main {
    max-width: 1000px; margin: auto; padding: 14px;
    display: grid; gap: 14px; grid-template-columns: minmax(0, 1fr);
}
@media (min-width: 860px) {
    main { grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); align-items: start; }
}
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 14px; min-width: 0; }
h2 { font-size: 16px; margin: 0 0 12px; }
.head-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 12px; }
.head-row h2 { margin: 0; }

button, .btn {
    border: 0; border-radius: 10px; padding: 12px 14px; min-height: 46px;
    font-size: 15px; font-weight: 600; color: var(--text); background: var(--btn);
    cursor: pointer; font-family: inherit;
    display: inline-flex; align-items: center; justify-content: center; text-align: center;
}
button:disabled { opacity: .5; cursor: not-allowed; }
.primary, .tab.active { background: var(--blue); color: #fff; }
.small { min-height: 34px; padding: 6px 10px; font-size: 13px; }
input[type=file] { display: none; }

.tabs { display: flex; gap: 8px; margin-bottom: 12px; }
.tabs .tab { flex: 1; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.actions > * { flex: 1 1 140px; }

.viewer {
    position: relative; width: 100%; aspect-ratio: 4 / 3;
    background: #050a12; border-radius: 10px; overflow: hidden;
}
.viewer > video, .viewer > img, .viewer > canvas {
    position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain;
}
.viewer > canvas { pointer-events: none; }
.placeholder {
    display: grid; place-items: center; height: 100%;
    color: #8090a9; text-align: center; padding: 20px; line-height: 1.5; font-size: 14px;
}
.guide {
    position: absolute; left: 5%; width: 90%;
    height: var(--gh, 25%); top: calc(50% - var(--gh, 25%) / 2);
    border: 2px solid #00e676; border-radius: 8px;
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, .5); pointer-events: none;
}
.guide::after {
    content: "Đặt ID vào trong khung"; position: absolute; left: 0; right: 0; top: -26px;
    text-align: center; font-size: 12px; color: #fff; text-shadow: 0 1px 3px #000;
}

.status { display: flex; align-items: center; gap: 8px; color: #aab8cc; font-size: 13px; margin: 12px 0 0; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: #6b7a93; flex: none; }
.dot[data-kind="ok"] { background: var(--green); }
.dot[data-kind="busy"] { background: var(--amber); }
.dot[data-kind="error"] { background: var(--red); }

.settings { display: grid; gap: 14px; margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--line); }
.field { display: flex; flex-direction: column; gap: 7px; font-size: 13px; color: var(--muted); }
.field b { color: var(--text); }
.field select, .field input[type=text] {
    background: var(--panel-2); color: var(--text); border: 1px solid var(--line);
    border-radius: 8px; padding: 10px; font-size: 16px; font-family: inherit;
}
.field input[type=range] { width: 100%; accent-color: var(--blue); }
.check { display: flex; align-items: flex-start; gap: 10px; font-size: 14px; color: var(--text); line-height: 1.4; }
.check input { width: 20px; height: 20px; margin-top: 1px; accent-color: var(--blue); flex: none; }
.check small { display: block; color: var(--muted); font-size: 12px; margin-top: 3px; }

.card { background: var(--panel-2); border: 1px solid #24344b; border-radius: 10px; padding: 12px; margin-bottom: 10px; }
.card-title { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
.card-id {
    font-size: 26px; font-weight: 700; overflow-wrap: anywhere; letter-spacing: .5px;
    font-family: ui-monospace, Consolas, "Courier New", monospace;
}
.card-conf { font-size: 13px; margin-top: 4px; color: var(--muted); }
.conf-high { color: var(--green); }
.conf-mid { color: var(--amber); }
.conf-low { color: var(--red); }
.meta { color: #8798b1; font-size: 12px; line-height: 1.5; margin: 0 0 12px; }

table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
th, td { text-align: left; padding: 6px 4px; border-bottom: 1px solid var(--line); overflow-wrap: anywhere; }
th { color: var(--muted); font-weight: 600; }

.feedback { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--line); }
.row { display: flex; gap: 8px; }
.row input { flex: 1; min-width: 0; }
.score { margin-top: 12px; font-size: 14px; line-height: 1.7; }
.note { color: #8798b1; font-size: 12px; line-height: 1.5; margin: 10px 0 0; }
</style>
</head>
<body>
<header>
    <h1>HRC Steel ID OCR</h1>
    <p id="modelInfo">Đang kiểm tra server...</p>
</header>

<main>
<section class="panel">
    <div class="tabs">
        <button id="tabPhoto" class="tab active">Chụp / chọn ảnh</button>
        <button id="tabLive" class="tab">Camera khung ngắm</button>
    </div>

    <div class="viewer" id="viewer">
        <div class="placeholder" id="placeholder"></div>
        <video id="video" autoplay playsinline muted hidden></video>
        <img id="shot" hidden alt="Ảnh đang nhận diện">
        <canvas id="overlay" hidden></canvas>
        <div class="guide" id="guide" hidden></div>
    </div>

    <div class="status">
        <span class="dot" id="statusDot" data-kind="idle"></span>
        <span id="statusText">Sẵn sàng.</span>
    </div>

    <div class="actions" id="photoActions">
        <label class="btn primary" for="camInput">📷 Chụp ảnh</label>
        <input id="camInput" type="file" accept="image/*" capture="environment">
        <label class="btn" for="galInput">🖼 Chọn từ thư viện</label>
        <input id="galInput" type="file" accept="image/*">
    </div>

    <div class="actions" id="liveActions" hidden>
        <button id="btnStart" class="primary">Bật camera</button>
        <button id="btnCapture" class="primary" disabled>Chụp &amp; nhận diện</button>
        <button id="btnResume" hidden>Chụp tiếp</button>
        <button id="btnStop" disabled>Tắt camera</button>
    </div>

    <div class="settings">
        <label class="field">
            <span>Model dùng để nhận diện</span>
            <select id="modelSelect">
                <option value="both">So sánh cả hai model</option>
                <option value="finetune">Chỉ model fine-tune</option>
                <option value="base">Chỉ model gốc</option>
            </select>
        </label>

        <label class="check">
            <input type="checkbox" id="skipDet">
            <span>Ảnh đã cắt sát dòng ID (bỏ bước dò vùng chữ)
                <small>Bật khi ảnh chỉ chứa dòng ID. Nếu để tắt mà không dò được chữ, server tự đọc cả ảnh.</small>
            </span>
        </label>

        <label class="field">
            <span>Ngưỡng tin cậy tối thiểu: <b id="thresholdValue">0%</b></span>
            <input id="threshold" type="range" min="0" max="95" step="5" value="0">
        </label>

        <label class="field" id="guideField" hidden>
            <span>Chiều cao khung ngắm: <b id="guideValue">25%</b></span>
            <input id="guideH" type="range" min="10" max="60" step="5" value="25">
        </label>
    </div>

    <p class="note">
        Chụp gần, đủ sáng, giữ chắc tay để chữ không bị mờ. Ảnh được thu nhỏ ngay trên điện thoại
        (tối đa 1600 px) trước khi gửi nên chạy nhanh qua Wi-Fi.
    </p>
</section>

<aside class="panel">
    <div class="head-row">
        <h2>Kết quả nhận diện</h2>
        <button id="btnRerun" class="small" hidden>Nhận diện lại ảnh này</button>
    </div>

    <p class="meta" id="meta"></p>
    <div id="results">Chưa có kết quả.</div>

    <div class="feedback" id="feedback" hidden>
        <label class="field">
            <span>ID đúng là (để chấm điểm model):</span>
            <div class="row">
                <input type="text" id="correct" autocomplete="off" autocapitalize="characters" spellcheck="false">
                <button id="btnFb" class="primary">Ghi nhận</button>
            </div>
        </label>
    </div>

    <div class="score" id="score"></div>
    <button id="btnResetScore" class="small" hidden style="margin-top:8px">Đặt lại điểm</button>
    <p class="note" id="scoreNote" hidden>
        Điểm tính theo kết quả tốt nhất của mỗi model (sau khi lọc theo ngưỡng tin cậy). So khớp chính xác từng ký tự,
        không tính khoảng trắng, phân biệt chữ hoa/thường.
    </p>
</aside>
</main>

<script>
const $ = (id) => document.getElementById(id);

const MAX_SEND = 1600;
const MODEL_ORDER = ["finetune", "base"];
const PLACEHOLDER = {
    photo: "Bấm “Chụp ảnh” để mở camera điện thoại, hoặc chọn ảnh từ thư viện.",
    live: "Bấm “Bật camera”, đặt dòng ID vào khung xanh rồi bấm “Chụp & nhận diện”. Chỉ phần trong khung được gửi đi."
};

const els = {
    viewer: $("viewer"), placeholder: $("placeholder"), video: $("video"), shot: $("shot"),
    overlay: $("overlay"), guide: $("guide"), dot: $("statusDot"), status: $("statusText")
};
const ctx = els.overlay.getContext("2d");

const state = {
    mode: "photo", stream: null, last: null, lastInfo: null, shotUrl: null,
    busy: false, feedbackSent: false,
    labels: { finetune: "Fine-tune", base: "Model gốc" },
    score: { finetune: { ok: 0, n: 0 }, base: { ok: 0, n: 0 } }
};

/* ---------- Tiện ích ---------- */

function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
}

function setStatus(text, kind = "idle") {
    els.status.textContent = text;
    els.dot.dataset.kind = kind;
}

function setAspect(w, h) { els.viewer.style.aspectRatio = `${w} / ${h}`; }

function showLayer(layer) {
    els.placeholder.hidden = layer !== "placeholder";
    els.video.hidden = layer !== "live";
    els.guide.hidden = layer !== "live";
    els.shot.hidden = layer !== "result";
    els.overlay.hidden = layer !== "result";
}

function showPlaceholder() {
    setAspect(4, 3);
    showLayer("placeholder");
}

function minScore() { return Number($("threshold").value) / 100; }
function filtered(list) { const m = minScore(); return (list || []).filter((d) => d.score >= m); }
function bestOf(list) { return list.reduce((b, d) => (!b || d.score > b.score ? d : b), null); }
function confClass(p) { return p >= 85 ? "conf-high" : p >= 60 ? "conf-mid" : "conf-low"; }

function buildTable(headers, rows) {
    const table = el("table");
    const head = el("tr");
    headers.forEach((h) => head.appendChild(el("th", "", h)));
    table.appendChild(el("thead")).appendChild(head);
    const body = el("tbody");
    rows.forEach((row) => {
        const tr = el("tr");
        row.forEach((cell) => tr.appendChild(el("td", "", cell)));
        body.appendChild(tr);
    });
    table.appendChild(body);
    return table;
}

/* ---------- Chế độ & trạng thái server ---------- */

function setMode(mode) {
    state.mode = mode;
    $("tabPhoto").classList.toggle("active", mode === "photo");
    $("tabLive").classList.toggle("active", mode === "live");
    $("photoActions").hidden = mode !== "photo";
    $("liveActions").hidden = mode !== "live";
    $("guideField").hidden = mode !== "live";
    els.placeholder.textContent = PLACEHOLDER[mode];
}

async function loadHealth() {
    try {
        const resp = await fetch("/health");
        const h = await resp.json();
        if (h.labels) state.labels = h.labels;

        const avail = h.models || {};
        const select = $("modelSelect");
        const both = avail.finetune && avail.base;

        select.querySelector('[value="both"]').disabled = !both;
        select.querySelector('[value="finetune"]').disabled = !avail.finetune;
        select.querySelector('[value="base"]').disabled = !avail.base;
        select.value = both ? "both" : (avail.finetune ? "finetune" : "base");

        const names = MODEL_ORDER.map((k) => `${state.labels[k] || k}: ${avail[k] ? "sẵn sàng" : "không có"}`);
        $("modelInfo").textContent = `${names.join(" · ")} · Thiết bị: ${h.device}`;
    } catch (error) {
        $("modelInfo").textContent = "Không kết nối được server.";
    }
}

/* ---------- Hiển thị kết quả ---------- */

function clearOverlay() { ctx.clearRect(0, 0, els.overlay.width, els.overlay.height); }

function drawBoxes() {
    clearOverlay();
    const data = state.last;
    if (!data) return;

    const key = MODEL_ORDER.find((k) => data.results[k]);
    if (!key) return;

    const W = data.image_width, H = data.image_height;
    const fs = Math.max(14, Math.round(W / 40));
    const list = filtered(data.results[key].detections);
    const best = bestOf(list);

    ctx.lineWidth = Math.max(2, W / 300);
    ctx.font = `bold ${fs}px Arial`;
    ctx.textBaseline = "middle";

    for (const d of list) {
        const [x1, y1, x2, y2] = d.bbox;
        if (x1 <= 1 && y1 <= 1 && x2 >= W - 1 && y2 >= H - 1) continue;   // vùng là cả ảnh: không vẽ khung

        const isBest = d === best;
        ctx.strokeStyle = isBest ? "#00ff83" : "#2ec4ff";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const label = `${d.text} ${d.confidence_percent}%`;
        const lh = fs + 10;
        const lw = ctx.measureText(label).width + 12;
        const ly = y1 - lh >= 0 ? y1 - lh : y1;
        const lx = Math.min(x1, Math.max(0, W - lw));

        ctx.fillStyle = isBest ? "#00a854" : "#1478a6";
        ctx.fillRect(lx, ly, lw, lh);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, lx + 6, ly + lh / 2);
    }
}

function renderResults() {
    const data = state.last;
    const box = $("results");

    if (!data) {
        box.textContent = "Chưa có kết quả.";
        $("meta").textContent = "";
        return;
    }

    let meta = `${data.image_width}×${data.image_height} px · ${Math.round(data.elapsed_ms)} ms`;
    if (data.skip_det) meta += " · đọc cả ảnh (đã bỏ dò vùng chữ)";
    else if (data.fallback_full_image) meta += " · không dò được vùng chữ nên đã đọc cả ảnh";
    $("meta").textContent = meta;

    const cards = [];
    for (const key of MODEL_ORDER) {
        const r = data.results[key];
        if (!r) continue;

        const list = filtered(r.detections);
        const top = bestOf(list);

        const card = el("div", "card");
        card.appendChild(el("div", "card-title", r.label));
        card.appendChild(el("div", "card-id", top ? top.text : "--"));
        card.appendChild(el(
            "div",
            "card-conf " + (top ? confClass(top.confidence_percent) : ""),
            top ? `Độ tin cậy ${top.confidence_percent}%` : "Không đọc được ID"
        ));

        if (list.length > 1) {
            const rows = [...list].sort((a, b) => b.score - a.score).slice(0, 10)
                .map((d) => [d.text, `${d.confidence_percent}%`]);
            card.appendChild(buildTable(["Vùng chữ khác", "Tin cậy"], rows));
        }
        cards.push(card);
    }
    box.replaceChildren(...cards);
}

function renderScore() {
    const lines = [];
    for (const key of MODEL_ORDER) {
        const s = state.score[key];
        if (!s.n) continue;
        const pct = Math.round((s.ok / s.n) * 100);
        lines.push(`${state.labels[key] || key}: đúng ${s.ok}/${s.n} (${pct}%)`);
    }
    $("score").replaceChildren(...lines.map((t) => el("div", "", t)));
    $("btnResetScore").hidden = !lines.length;
    $("scoreNote").hidden = !lines.length;
}

async function sendFeedback() {
    const data = state.last;
    if (!data || state.feedbackSent) return;

    const correct = $("correct").value.trim();
    if (!correct) { setStatus("Hãy nhập ID đúng trước khi ghi nhận.", "error"); return; }

    const fd = new FormData();
    fd.append("id", data.id);
    fd.append("correct_text", correct);

    try {
        const resp = await fetch("/ocr/feedback", { method: "POST", body: fd });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    } catch (error) {
        setStatus("Không ghi nhận được: " + error.message, "error");
        return;
    }

    const norm = (s) => s.replace(/\s+/g, "");
    for (const key of MODEL_ORDER) {
        const r = data.results[key];
        if (!r) continue;
        const top = bestOf(filtered(r.detections));
        state.score[key].n += 1;
        if (top && norm(top.text) === norm(correct)) state.score[key].ok += 1;
    }

    state.feedbackSent = true;
    $("btnFb").disabled = true;
    renderScore();
    setStatus("Đã ghi nhận ID đúng.", "ok");
}

/* ---------- Gửi ảnh lên server ---------- */

function renderToBlob(source, sx, sy, sw, sh) {
    const scale = Math.min(1, MAX_SEND / Math.max(sw, sh));
    const w = Math.max(1, Math.round(sw * scale));
    const h = Math.max(1, Math.round(sh * scale));

    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").drawImage(source, sx, sy, sw, sh, 0, 0, w, h);

    return new Promise((resolve, reject) => {
        canvas.toBlob(
            (blob) => (blob ? resolve({ blob, w, h }) : reject(new Error("Không mã hóa được ảnh"))),
            "image/jpeg", 0.9
        );
    });
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("Không đọc được ảnh"));
        img.src = url;
    });
}

async function sendImage(blob) {
    const fd = new FormData();
    fd.append("file", blob, "capture.jpg");
    fd.append("models", $("modelSelect").value);
    fd.append("skip_det", $("skipDet").checked ? "true" : "false");

    const resp = await fetch("/ocr/predict", { method: "POST", body: fd });

    let data = null;
    try { data = await resp.json(); } catch (error) { data = null; }

    if (!resp.ok) throw new Error((data && data.detail) || `HTTP ${resp.status}`);
    return data;
}

async function runOcr(info) {
    if (state.busy) return;
    state.busy = true;

    state.lastInfo = info;
    state.last = null;
    state.feedbackSent = false;
    clearOverlay();
    renderResults();
    $("feedback").hidden = true;
    $("btnRerun").hidden = true;

    if (state.shotUrl) URL.revokeObjectURL(state.shotUrl);
    state.shotUrl = URL.createObjectURL(info.blob);
    els.shot.src = state.shotUrl;
    setAspect(info.w, info.h);
    showLayer("result");

    if (state.mode === "live") {
        $("btnCapture").disabled = true;
        $("btnResume").hidden = false;
    }

    setStatus("Đang nhận diện...", "busy");

    try {
        const data = await sendImage(info.blob);
        state.last = data;

        els.overlay.width = data.image_width;
        els.overlay.height = data.image_height;
        drawBoxes();
        renderResults();

        $("correct").value = "";
        $("btnFb").disabled = false;
        $("feedback").hidden = false;
        $("btnRerun").hidden = false;

        const total = Object.values(data.results).reduce((n, r) => n + r.detections.length, 0);
        if (total) setStatus(`Xong trong ${Math.round(data.elapsed_ms)} ms.`, "ok");
        else setStatus("Không đọc được ID nào. Thử chụp gần hơn, đủ sáng, hoặc bật “Ảnh đã cắt sát ID”.", "idle");
    } catch (error) {
        setStatus("Lỗi: " + error.message, "error");
    } finally {
        state.busy = false;
    }
}

/* ---------- Chụp / chọn ảnh ---------- */

async function handleFile(input) {
    const file = input.files[0];
    input.value = "";
    if (!file || state.busy) return;

    setStatus("Đang chuẩn bị ảnh...", "busy");
    const url = URL.createObjectURL(file);

    try {
        const img = await loadImage(url);
        const info = await renderToBlob(img, 0, 0, img.naturalWidth, img.naturalHeight);
        await runOcr(info);
    } catch (error) {
        setStatus("Lỗi: " + error.message, "error");
    } finally {
        URL.revokeObjectURL(url);
    }
}

$("camInput").addEventListener("change", (e) => handleFile(e.target));
$("galInput").addEventListener("change", (e) => handleFile(e.target));

/* ---------- Camera khung ngắm ---------- */

function updateGuide() {
    const value = Number($("guideH").value);
    $("guideValue").textContent = value + "%";
    els.guide.style.setProperty("--gh", value + "%");
}

function syncVideoAspect() {
    if (els.video.videoWidth) setAspect(els.video.videoWidth, els.video.videoHeight);
}

async function startCamera() {
    if (state.stream) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("Camera trực tiếp cần HTTPS hoặc localhost. Dùng nút “Chụp ảnh” ở tab bên, hoặc mở trang bằng liên kết HTTPS.", "error");
        return;
    }

    setStatus("Đang xin quyền camera...", "busy");

    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } }
        });
        els.video.srcObject = state.stream;
        await els.video.play();
    } catch (error) {
        if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
        state.stream = null;
        setStatus("Không mở được camera: " + error.message, "error");
        return;
    }

    syncVideoAspect();
    showLayer("live");
    $("btnStart").disabled = true;
    $("btnCapture").disabled = false;
    $("btnStop").disabled = false;
    $("btnResume").hidden = true;
    setStatus("Camera đã bật. Đặt ID vào khung rồi bấm “Chụp & nhận diện”.", "ok");
}

function stopCamera() {
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    els.video.srcObject = null;
    $("btnStart").disabled = false;
    $("btnCapture").disabled = true;
    $("btnStop").disabled = true;
    $("btnResume").hidden = true;
}

function resumeLive() {
    if (!state.stream) return;
    syncVideoAspect();
    showLayer("live");
    $("btnResume").hidden = true;
    $("btnCapture").disabled = false;
    setStatus("Camera đang chờ chụp.", "ok");
}

async function captureFromGuide() {
    const v = els.video;
    if (state.busy || !state.stream || v.readyState < 2 || !v.videoWidth) return;

    // Chỉ cắt phần nằm trong khung ngắm (rộng 90%, cao theo thanh trượt, nằm giữa).
    const vw = v.videoWidth, vh = v.videoHeight;
    const gw = 0.9, gh = Number($("guideH").value) / 100;
    const sw = vw * gw, sh = vh * gh;
    const sx = (vw - sw) / 2, sy = (vh - sh) / 2;

    try {
        const info = await renderToBlob(v, sx, sy, sw, sh);
        await runOcr(info);
    } catch (error) {
        setStatus("Lỗi: " + error.message, "error");
    }
}

/* ---------- Sự kiện ---------- */

$("tabPhoto").addEventListener("click", () => {
    stopCamera();
    setMode("photo");
    if (state.lastInfo) { setAspect(state.lastInfo.w, state.lastInfo.h); showLayer("result"); }
    else showPlaceholder();
    setStatus("Chụp hoặc chọn ảnh để nhận diện.", "idle");
});

$("tabLive").addEventListener("click", () => {
    if (state.mode === "live") return;
    setMode("live");
    if (state.lastInfo) { setAspect(state.lastInfo.w, state.lastInfo.h); showLayer("result"); }
    else showPlaceholder();
    setStatus("Bấm “Bật camera” để bắt đầu.", "idle");
});

$("btnStart").addEventListener("click", startCamera);
$("btnCapture").addEventListener("click", captureFromGuide);
$("btnResume").addEventListener("click", resumeLive);
$("btnStop").addEventListener("click", () => {
    stopCamera();
    if (state.lastInfo) { setAspect(state.lastInfo.w, state.lastInfo.h); showLayer("result"); }
    else showPlaceholder();
    setStatus("Đã tắt camera.", "idle");
});

$("btnRerun").addEventListener("click", () => { if (state.lastInfo) runOcr(state.lastInfo); });
$("btnFb").addEventListener("click", sendFeedback);
$("correct").addEventListener("keydown", (e) => { if (e.key === "Enter") sendFeedback(); });
$("btnResetScore").addEventListener("click", () => {
    state.score = { finetune: { ok: 0, n: 0 }, base: { ok: 0, n: 0 } };
    renderScore();
});

$("threshold").addEventListener("input", () => {
    $("thresholdValue").textContent = $("threshold").value + "%";
    if (state.last) { drawBoxes(); renderResults(); }
});
$("guideH").addEventListener("input", updateGuide);

els.video.addEventListener("loadedmetadata", syncVideoAspect);
window.addEventListener("beforeunload", stopCamera);

setMode("photo");
showPlaceholder();
updateGuide();
renderScore();
loadHealth();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML)


# =========================
# CHẠY SERVER
# =========================
def lan_ip():
    """Lấy IP của máy trong mạng nội bộ (không gửi dữ liệu ra ngoài)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="HRC Steel ID OCR - Phone Demo")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--https", action="store_true", help="Dùng cert.pem và key.pem cạnh file này.")
    args = parser.parse_args()

    ssl_kwargs = {}
    scheme = "http"

    if args.https:
        cert, key = BASE_DIR / "cert.pem", BASE_DIR / "key.pem"
        if not cert.exists() or not key.exists():
            raise SystemExit("Chưa có cert.pem / key.pem cạnh demo_phone.py (xem hướng dẫn tạo chứng chỉ).")
        ssl_kwargs = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        scheme = "https"

    print(f"\nTrên máy tính:  {scheme}://localhost:{args.port}")
    print(f"Trên điện thoại (cùng Wi-Fi):  {scheme}://{lan_ip()}:{args.port}\n")

    uvicorn.run(app, host="0.0.0.0", port=args.port, **ssl_kwargs)