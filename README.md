# AI Object Detection Microservice

A microservice architecture for object detection using YOLO and FastAPI with a beautiful web interface.

add images in folder to test

## Features

- Modern, responsive web interface
- Real-time object detection using YOLO
- Microservice architecture
- Docker containerization
- JSON and image outputs
- Professional UI with Bootstrap 5

## Quick Start

1. Clone the repository
2. Run: `docker-compose up --build`
3. Open: http://localhost:8000
4. Upload an image and detect objects!

## Architecture

- **UI Backend**: FastAPI service with web interface (Port 8000)
- **AI Backend**: Object detection service with YOLO (Port 8001)
- **Communication**: RESTful API between services

## API Endpoints

- `GET /` - Web interface
- `POST /upload` - Upload image for detection
- `GET /health` - Health check
