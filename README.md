AI Object Detection Microservice
A modern AI object detection microservice built with FastAPI, supporting multiple YOLO models. Features a beautiful web interface with drag-and-drop upload, real-time detection, and downloadable results.


Multiple AI Models: YOLOv8, YOLOv9, YOLOv3, MobileNet-SSD

CPU-Only: No GPU required, runs on any machine

Web Interface: Drag-and-drop image upload

Real-time Results: Instant object detection with bounding boxes

Download Results: JSON data and annotated images


# 🐳 Quick Start with Docker (Recommended)
Prerequisites
Docker and Docker Compose

4GB+ RAM

1. Clone Repository
bash
git clone https://github.com/YOUR_USERNAME/ai-object-detection-microservice.git
cd ai-object-detection-microservice
2. Start Application
bash
# Start all services
docker compose up --build

# Or run in background
docker compose up --build -d
3. Access Application
Web Interface: http://localhost:8000

AI Backend API: http://localhost:8001

4. Stop Application
bash
docker compose down



# 💻 Manual Setup (Without Docker)
Prerequisites
Python 3.11+

4GB+ RAM

1. Clone and Setup
bash
git clone https://github.com/YOUR_USERNAME/ai-object-detection-microservice.git
cd ai-object-detection-microservice
2. Install AI Backend
bash
cd ai_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start AI backend
python main.py
3. Install UI Backend (New Terminal)
bash
cd ui_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start UI backend
python main.py
4. Access Application
Web Interface: http://localhost:8000

AI Backend: http://localhost:8001

📖 How to Use
Open Browser: Go to http://localhost:8000

Upload Image: Drag & drop or click to select an image

Select Model: Choose YOLOv8, YOLOv9, YOLOv3, or MobileNet-SSD

Set Confidence: Adjust detection threshold (0.1-0.9)

Detect Objects: Click "Detect Objects" button

View Results: See detected objects with bounding boxes

Download: Save JSON results 

🔧 API Endpoints
AI Backend (Port 8001)
GET /health - Health check

POST /detect - Upload image for detection

GET /outputs/images/{filename} - Download result images

UI Backend (Port 8000)
GET / - Web interface

POST /upload - Upload proxy to AI backend

GET /health - Health check

📋 Requirements
System Requirements
RAM: 4GB minimum (2GB for Docker, 2GB for models)

Storage: 2GB free space

CPU: Any modern processor

OS: Linux, macOS, Windows

Python Dependencies
See requirements.txt files in each service directory.