from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import base64
import cv2
import numpy as np
from datetime import datetime
import hashlib
try:
    from detection import AttentionMonitor
    DETECTION_AVAILABLE = True
except Exception as e:
    print(f"Warning: Detection module not available: {e}")
    DETECTION_AVAILABLE = False
    AttentionMonitor = None
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
from contextlib import contextmanager


class UserRegistration(BaseModel):
    name: str
    email: str
    password: str  
    enroll: str    

class UserLogin(BaseModel):
    email: str
    password: str  

class ExamSubmission(BaseModel):
    answers: List[dict]
    auto: bool = False


app = FastAPI(title="ProctorVision API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


monitor = AttentionMonitor() if DETECTION_AVAILABLE else None


DATABASE_FILE = "proctordb.sqlite"

def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            enrollment TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_registrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            enrollment TEXT NOT NULL,
            face_image_path TEXT NOT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (enrollment) REFERENCES users(enrollment)
        )
    ''')
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            enrollment TEXT NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            submitted_at TIMESTAMP,
            answers TEXT,  -- JSON string
            violations TEXT,  -- JSON string of detected violations
            auto_submitted BOOLEAN DEFAULT FALSE
        )
    ''')
    
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            violation_type TEXT NOT NULL,
            description TEXT,
            screenshot_path TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES exam_sessions(id)
        )
    ''')
    
    conn.commit()
    conn.close()

@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row  
    try:
        yield conn
    finally:
        conn.close()

def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


os.makedirs("faces", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
os.makedirs("static", exist_ok=True)


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_database()
    print("✅ Database initialized")
    print("✅ ProctorVision API Server started")

@app.get("/")
async def serve_frontend():
    """Serve the main HTML file."""
    return FileResponse("ProctorVision.html")

@app.post("/api/register")
async def register_user(user: UserRegistration):
    """Register a new user."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            
            cursor.execute("SELECT id FROM users WHERE email = ? OR enrollment = ?", 
                         (user.email, user.enroll))
            if cursor.fetchone():
                raise HTTPException(status_code=400, detail="User with this email or enrollment already exists")
            
            
            password_hash = hash_password(user.password)
            cursor.execute('''
                INSERT INTO users (name, email, password_hash, enrollment)
                VALUES (?, ?, ?, ?)
            ''', (user.name, user.email, password_hash, user.enroll))
            
            conn.commit()
            return {"message": "User registered successfully", "enrollment": user.enroll}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/login")
async def login_user(credentials: UserLogin):
    """Login user with email and password."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            password_hash = hash_password(credentials.password)
            cursor.execute('''
                SELECT id, name, enrollment FROM users 
                WHERE email = ? AND password_hash = ?
            ''', (credentials.email, password_hash))
            
            user = cursor.fetchone()
            if not user:
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            return {
                "message": "Login successful",
                "user": {
                    "id": user["id"],
                    "name": user["name"],
                    "enrollment": user["enrollment"]
                }
            }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/check_face")
async def check_face(enroll: str):
    """Check if user has registered face."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM face_registrations WHERE enrollment = ?", (enroll,))
            face_exists = cursor.fetchone() is not None
            
            return {"hasFace": face_exists, "enrollment": enroll}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/register_face")
async def register_face(
    enroll: str = Form(...),  
    image: UploadFile = File(...)
):
    """Register user's face image."""
    try:
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE enrollment = ?", (enroll,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="User not found")
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_{enroll}_{timestamp}.jpg"
        filepath = os.path.join("faces", filename)
        
        
        contents = await image.read()
        with open(filepath, "wb") as f:
            f.write(contents)
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM face_registrations WHERE enrollment = ?", (enroll,))
            
            cursor.execute('''
                INSERT INTO face_registrations (enrollment, face_image_path)
                VALUES (?, ?)
            ''', (enroll, filepath))
            conn.commit()
        
        return {"message": "Face registered successfully", "filename": filename}
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect")
async def detect_violations(frame: UploadFile = File(...)):
    """Process frame for violation detection."""
    try:
        if not DETECTION_AVAILABLE:
            return {
                "warnings": ["Detection system not available"],
                "head_status": "unknown",
                "hand_count": 0,
                "gadgets": [],
                "screenshot_path": None
            }
            
        
        contents = await frame.read()
        nparr = np.frombuffer(contents, np.uint8)
        cv_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cv_frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        
        warnings, head_status, hand_count, gadgets = monitor.process_frame(cv_frame)
        
        
        screenshot_path = None
        if warnings or gadgets:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join("screenshots", f"violation_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, cv_frame)
        
        return {
            "warnings": warnings,
            "head_status": head_status,
            "hand_count": hand_count,
            "gadgets": gadgets,
            "timestamp": datetime.now().isoformat(),
            "screenshot_saved": screenshot_path is not None
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/api/start_exam")
async def start_exam(enrollment: str = Form(...)):
    """Start a new exam session."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            
            cursor.execute("SELECT id FROM users WHERE enrollment = ?", (enrollment,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail="User not found")
            
            
            cursor.execute('''
                INSERT INTO exam_sessions (enrollment)
                VALUES (?)
            ''', (enrollment,))
            
            session_id = cursor.lastrowid
            conn.commit()
            
            return {
                "message": "Exam session started",
                "session_id": session_id,
                "started_at": datetime.now().isoformat()
            }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/submit_exam")
async def submit_exam(submission: ExamSubmission):
    """Submit exam answers."""
    try:
        timestamp = datetime.now()
        
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            
            cursor.execute('''
                INSERT INTO exam_sessions (enrollment, submitted_at, answers, auto_submitted)
                VALUES (?, ?, ?, ?)
            ''', ("TEMP_SESSION", timestamp, json.dumps(submission.answers), submission.auto))
            
            session_id = cursor.lastrowid
            conn.commit()
        
        return {
            "message": "Exam submitted successfully",
            "session_id": session_id,
            "auto_submitted": submission.auto
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "ProctorVision API is running"
    }

@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            
            
            cursor.execute("SELECT COUNT(*) as count FROM users")
            user_count = cursor.fetchone()["count"]
            
            
            cursor.execute("SELECT COUNT(*) as count FROM face_registrations")
            face_count = cursor.fetchone()["count"]
            
            
            cursor.execute("SELECT COUNT(*) as count FROM exam_sessions")
            exam_count = cursor.fetchone()["count"]
            
            return {
                "users_registered": user_count,
                "faces_registered": face_count,
                "exam_sessions": exam_count,
                "system_status": "operational"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(" Starting ProctorVision API Server...")
    print(" Frontend will be available at: http://localhost:8002")
    print(" API documentation at: http://localhost:8002/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )