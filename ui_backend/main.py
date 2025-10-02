    
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import httpx
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

app = FastAPI(title="UI Backend - Object Detection Interface", version="2.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# AI Backend URL using Docker service name for internal communication
AI_BACKEND_URL = os.getenv("AI_BACKEND_URL", "http://ai_backend:8001")

@app.get("/", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    """Serve the main upload interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_and_detect(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("yolov8"),
    conf: float = Form(0.25)
):
    """Proxy file upload to AI backend and return results with corrected URLs"""
    try:
        # Prepare the multipart form data for AI backend
        files = {"file": (file.filename, await file.read(), file.content_type)}
        data = {"model": model, "conf": conf}
        
        # Forward request to AI backend using Docker service name
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{AI_BACKEND_URL}/detect",
                files=files,
                data=data
            )
            
        if response.status_code == 200:
            result = response.json()
            
            # Fix the image URL to be accessible from browser
            # Replace docker internal URL with localhost URL for frontend
            if "public_image_url" in result:
                # Convert ai_backend:8001 to localhost:8001 for browser access
                result["public_image_url"] = result["public_image_url"].replace(
                    "ai_backend:8001", "localhost:8001"
                )
            
            # Also provide a relative URL that goes through this backend
            if "image_filename" in result:
                result["ui_image_url"] = f"/proxy-image/{result['image_filename']}"
            
            return JSONResponse(content=result)
        else:
            error_detail = response.text
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"AI backend error: {error_detail}"
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI backend timeout - processing took too long")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot connect to AI backend")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@app.get("/proxy-image/{image_filename}")
async def proxy_image(image_filename: str):
    """Proxy image requests to AI backend to avoid CORS issues"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{AI_BACKEND_URL}/outputs/images/{image_filename}")
            
        if response.status_code == 200:
            return JSONResponse(
                content=response.content,
                headers={
                    "Content-Type": "image/jpeg",
                    "Cache-Control": "public, max-age=3600"
                }
            )
        else:
            raise HTTPException(status_code=404, detail="Image not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error proxying image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "UI Backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
