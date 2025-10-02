import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import json
from datetime import datetime
import uvicorn
from ultralytics import YOLO
import torch
import uuid
from PIL import Image

try:
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel
    add_safe_globals([DetectionModel])
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
os.makedirs(BASE_DIR / "output" / "images", exist_ok=True)
os.makedirs(BASE_DIR / "output" / "json", exist_ok=True)

MSSD_DIR = Path(os.getenv("MOBILENETSSD_DIR", BASE_DIR / "models" / "mobilenetssd"))
MODEL_CACHE = {"yolov8": None, "yolov9": None, "yolov3": None, "mobilenetssd": None}

def get_yolo(model_key: str):
    if MODEL_CACHE[model_key] is None:
        if model_key == "yolov8":
            MODEL_CACHE[model_key] = YOLO("yolov8n.pt")
        elif model_key == "yolov9":
            MODEL_CACHE[model_key] = YOLO("yolov9c.pt")
        elif model_key == "yolov3":
            MODEL_CACHE[model_key] = YOLO("yolov3u.pt")
        try:
            MODEL_CACHE[model_key].to("cpu")
        except Exception:
            pass
    return MODEL_CACHE[model_key]

def resolve_mssd_paths():
    """Find MobileNet-SSD prototxt and caffemodel files"""
    proto_names = [
        "MobileNetSSD_deploy.prototxt",
        "deploy.prototxt",
        "MobileNetSSD_deploy.prototxt.txt",
    ]
    model_names = [
        "MobileNetSSD_deploy.caffemodel",
    ]

    search_dirs = [MSSD_DIR]
    default_models_dir = BASE_DIR / "models"
    if default_models_dir not in search_dirs:
        search_dirs.append(default_models_dir)

    found_proto = None
    found_model = None
    for d in search_dirs:
        if not d.exists():
            continue
        for name in proto_names:
            p = d / name
            if p.exists():
                found_proto = p
                break
        for name in model_names:
            m = d / name
            if m.exists():
                found_model = m
                break
        if found_proto and found_model:
            return found_proto, found_model

    # If files not found, raise a more helpful error
    searched_locations = []
    for d in search_dirs:
        searched_locations.extend([
            str(d / proto_names[0]),
            str(d / proto_names[1]), 
            str(d / model_names[0])
        ])
    
    raise FileNotFoundError(
        f"MobileNet-SSD model files not found. Please download:\n"
        f"1. MobileNetSSD_deploy.prototxt\n"
        f"2. MobileNetSSD_deploy.caffemodel\n"
        f"And place them in: {MSSD_DIR}\n"
        f"Download from: https://github.com/chuanqi305/MobileNet-SSD\n"
        f"Searched in: {'; '.join(searched_locations)}"
    )

def get_mobilenet_ssd():
    """Load MobileNet-SSD via OpenCV DNN"""
    try:
        proto, weights = resolve_mssd_paths()
        net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except FileNotFoundError as e:
        # Return None so we can handle this gracefully
        print(f"MobileNet-SSD setup failed: {e}")
        return None

MSSD_CLASSES = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
    "cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MODEL_CACHE["yolov8"] = YOLO("yolov8n.pt")
        try:
            MODEL_CACHE["yolov8"].to("cpu")
        except Exception:
            pass
        print("Warm-up: YOLOv8n loaded on CPU")
    except Exception as e:
        print(f"Warm-up skipped: {e}")
    yield

app = FastAPI(title="AI Backend - Object Detection (CPU)", version="3.3.0", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "output")), name="outputs")

@app.get("/")
def root():
    return {"service": "AI Backend (CPU)", "endpoints": ["/health", "/assets", "/detect", "/outputs/*"]}

@app.get("/health")
async def health_check():
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    return {
        "status": "healthy",
        "service": "AI Backend",
        "model_ready": MODEL_CACHE["yolov8"] is not None,
        "cuda_available": cuda
    }

@app.get("/assets")
def assets():
    """Report MobileNet-SSD asset resolution for debugging"""
    try:
        proto, model = resolve_mssd_paths()
        return {
            "mobilenetssd_dir": str(MSSD_DIR),
            "resolved_prototxt": str(proto),
            "resolved_caffemodel": str(model),
            "prototxt_exists": proto.exists(),
            "caffemodel_exists": model.exists()
        }
    except Exception as e:
        return {
            "mobilenetssd_dir": str(MSSD_DIR),
            "error": str(e),
            "note": "MobileNet-SSD files not found. Download from: https://github.com/chuanqi305/MobileNet-SSD"
        }

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model: str = Form("yolov8"),
    conf: float = Form(0.25)
):
    model = (model or "yolov8").lower()
    if model not in ("yolov8", "yolov9", "yolov3", "mobilenetssd"):
        raise HTTPException(status_code=400, detail="Invalid model; choose yolov8, yolov9, yolov3, or mobilenetssd")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img = np.array(image)

        detections = []
        if model == "mobilenetssd":
            # Handle MobileNet-SSD with better error handling
            if MODEL_CACHE["mobilenetssd"] is None:
                net = get_mobilenet_ssd()
                if net is None:
                    raise HTTPException(
                        status_code=424, 
                        detail="MobileNet-SSD model files not found. Please download MobileNetSSD_deploy.prototxt and MobileNetSSD_deploy.caffemodel from https://github.com/chuanqi305/MobileNet-SSD and place them in the models/mobilenetssd/ directory."
                    )
                MODEL_CACHE["mobilenetssd"] = net
            
            net = MODEL_CACHE["mobilenetssd"]
            blob = cv2.dnn.blobFromImage(
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                scalefactor=1/127.5, size=(300, 300), mean=(127.5, 127.5, 127.5),
                swapRB=False, crop=False
            )
            net.setInput(blob)
            out = net.forward()
            h, w = img.shape[:2]
            for i in range(out.shape[2]):
                score = float(out[0, 0, i, 2])
                if score < conf:
                    continue
                cls_id = int(out[0, 0, i, 1])
                x1 = int(out[0, 0, i, 3] * w)
                y1 = int(out[0, 0, i, 4] * h)
                x2 = int(out[0, 0, i, 5] * w)
                y2 = int(out[0, 0, i, 6] * h)
                cls_name = MSSD_CLASSES[cls_id] if 0 <= cls_id < len(MSSD_CLASSES) else str(cls_id)
                detections.append({"class": cls_name, "confidence": round(score, 3),
                                   "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}})
        else:
            # All YOLO models use Ultralytics API
            ym = get_yolo(model)
            results = ym.predict(source=img, device="cpu", conf=conf, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = ym.names.get(cls_id, str(cls_id))
                    detections.append({"class": cls_name, "confidence": round(score, 3),
                                       "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}})

        # Annotate and persist
        uid = str(uuid.uuid4())[:8]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated = img.copy()
        for d in detections:
            b = d["bbox"]
            cv2.rectangle(annotated, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
            cv2.putText(annotated, f"{d['class']} ({d['confidence']:.2f})",
                        (b["x1"], max(0, b["y1"] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        img_name = f"{ts}_{uid}_{model}_detected.jpg"
        json_name = f"{ts}_{uid}_{model}_results.json"
        out_img_path = BASE_DIR / "output" / "images" / img_name
        out_json_path = BASE_DIR / "output" / "json" / json_name

        cv2.imwrite(str(out_img_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        payload = {
            "timestamp": ts,
            "image_id": uid,
            "model": model,
            "confidence_threshold": conf,
            "detections": detections,
            "total_objects": len(detections),
            "output_image": str(out_img_path.relative_to(BASE_DIR)),
            "public_image_url": f"http://localhost:8001/outputs/images/{img_name}",
            "image_filename": img_name
        }
        with open(out_json_path, "w") as f:
            json.dump(payload, f, indent=2)
        return JSONResponse(content=payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
