#  ProctorVision AI - Advanced Exam Proctoring System

A complete AI-powered exam proctoring system with real-time monitoring, face detection, audio analysis, and comprehensive violation tracking. Built with FastAPI backend, intelligent computer vision, and modern web technologies.

##  Key Features

###  **Visual Monitoring**
- **Face Detection & Tracking** - Real-time face mesh analysis
- **Head Position Monitoring** - Detects looking away from camera
- **Hand Visibility Detection** - Ensures both hands remain visible
- **Enhanced Object Detection** - YOLOv8 detects 17+ prohibited items
- **Smart Screenshot Capture** - Auto-saves violation evidence

###  **Audio Monitoring** 
- **Background Speech Detection** - Identifies conversations/assistance
- **Multi-frequency Analysis** - Advanced voice pattern recognition
- **Adjustable Sensitivity** - Customizable detection thresholds
- **Silent Monitoring** - Analyzes audio without playback feedback

###  **Alert System**
- **Visual Warnings** - Real-time on-screen notifications
- **Audio Alerts** - Sound notifications for violations
- **Progressive Escalation** - Warnings intensify with repeated violations
- **Smart Categorization** - Different alerts for different violation types

###  **Web Interface**
- **Complete User Flow** - Registration → Login → Face Setup → Exam
- **Responsive Design** - Works on desktop and mobile devices
- **Real-time Status** - Live monitoring indicators and timers
- **Guest Mode** - Quick access for testing and demonstrations

###  **Database System**
- **users** - User accounts with secure password hashing
- **face_registrations** - Biometric face data storage
- **exam_sessions** - Complete exam tracking with answers
- **violations** - Detailed violation logs with timestamps

###  **Advanced Detection**
- **17+ Object Types** - Phones, tablets, laptops, books, calculators, cameras, smart watches, headphones, etc.
- **Intelligent Filtering** - Reduces false positives from normal movements
- **Contextual Warnings** - Specific messages for different violation types
- **Performance Optimized** - 2-second detection intervals for responsiveness

##  Project Structure

```
ProctorVision/
├── backend_server.py         # FastAPI backend with all endpoints
├── detection.py             # AI detection engine (OpenCV + YOLO + MediaPipe)
├── ProctorVision.html       # Complete frontend SPA with audio integration
├── run_frontend.py          # Frontend development server
├── start_proctorVision.bat  # One-click startup script
├── requirements.txt         # Python dependencies list
├── yolov8n.pt              # YOLOv8 model weights
├── proctordb.sqlite        # SQLite database (auto-created)
├── faces/                  # Face registration images
├── screenshots/           # Violation evidence screenshots
└── static/               # Static web assets
```

##  Quick Start Guide

###  **One-Click Setup (Recommended)**
```cmd
# Double-click this file or run in terminal:
start_proctorVision.bat
```
This automatically starts both backend and frontend servers!

###  **Manual Setup**

#### 1. **Install Dependencies**
```cmd
# Install required Python packages
pip install fastapi uvicorn opencv-python mediapipe ultralytics numpy python-multipart
```

#### 2. **Start Backend Server**
```cmd
# Start the FastAPI backend (runs on port 8002)
python backend_server.py
```

#### 3. **Start Frontend Server** (New Terminal)
```cmd
# Start the development frontend server (runs on port 3000)
python run_frontend.py
```

#### 4. **Access Application**
Open your browser and navigate to: **http://localhost:3000/ProctorVision.html**

##  Application URLs

- **Main Application**: http://localhost:3000/ProctorVision.html
- **Backend API**: http://localhost:8002
- **API Documentation**: http://localhost:8002/docs
- **Health Check**: http://localhost:8002/api/health

##  System Requirements

### **Minimum Requirements:**
- **OS**: Windows 10/11, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Camera**: Webcam with 720p or higher resolution
- **Microphone**: Built-in or external microphone
- **Browser**: Chrome 80+, Firefox 75+, Safari 13+, Edge 80+

### **Recommended Setup:**
- **CPU**: Intel i5 or AMD Ryzen 5 (for real-time AI processing)
- **GPU**: Optional but improves YOLOv8 performance
- **Internet**: Stable connection for model downloads
- **Lighting**: Good room lighting for optimal face detection

##  Complete API Reference

###  **User Management**
```http
POST /api/register          # Register new user account
POST /api/login             # Authenticate user login
GET  /api/check_face        # Check if user has face registered
POST /api/register_face     # Upload and register face biometrics
```

###  **Exam Operations**
```http
POST /api/start_exam        # Initialize new exam session
POST /api/submit_exam       # Submit completed exam answers
POST /api/detect            # Real-time violation detection
GET  /api/exam_status       # Get current exam session status
```

###  **System & Analytics**
```http
GET  /api/health           # System health and status check
GET  /api/stats            # User and exam statistics
GET  /api/violations       # Violation history and logs
```

###  **Detection Capabilities**
The `/api/detect` endpoint processes camera frames and returns:
```json
{
  "warnings": ["Array of violation messages"],
  "head_status": "forward/left/right",
  "hand_count": 0-2,
  "gadgets": ["Array of detected objects"],
  "screenshot_path": "Path to saved evidence"
}
```

## � Complete User Guide

###  **Step 1: User Registration**
1. Open **http://localhost:3000/ProctorVision.html**
2. Click **"Get Started"**
3. Fill in registration form:
   - Name, Enrollment Number, Password
4. Click **"Register"** to create account

###  **Step 2: Face Registration** 
1. Click **"Face Register"** tab
2. **Allow camera access** when prompted
3. **Allow/deny microphone** (both work fine)
4. Position face clearly in camera frame
5. Click **"Capture Photo"** when ready
6. Enter your **enrollment number**
7. Click **"Upload & Register Face"**

###  **Step 3: Take Exam**
1. Click **"Take Exam"** tab
2. **Automatic monitoring starts**:
   -  3-hour timer begins
   -  Camera monitoring active
   -  Audio monitoring active (if enabled)
   -  AI detection running every 2 seconds

3. **Follow exam rules**:
   - Keep both hands visible on desk
   - Look straight at camera
   - Maintain quiet environment
   - No prohibited devices

4. **Answer questions** and navigate with Previous/Next
5. **Submit exam** when complete

###  **Quick Test Mode**
- Use **Guest Login** for immediate testing
- Click **"Cancel"** when prompted for authentication
- Bypasses registration for quick demos

## �️ Advanced Monitoring System

###  **Visual Detection**
- **Face Tracking**: MediaPipe face mesh with 468 landmarks
- **Head Position**: Precise left/right/forward detection
- **Hand Detection**: Counts and tracks both hands continuously  
- **Object Recognition**: YOLOv8-powered detection of:
  ```
   Mobile phones      Laptops            Books
   Notebooks          Calculators        Cameras  
   Headphones         Smart watches      Computer mice
   Keyboards          Game controllers   Bottles
   Tablets            Credit cards       Clipboards
  ```

###  **Audio Analysis** 
- **Speech Detection**: Multi-frequency voice pattern analysis
- **Background Monitoring**: Identifies conversations and assistance
- **Silent Operation**: Monitors without audio feedback
- **Adjustable Sensitivity**: 10-50 range for different environments

###  **Smart Alert System**
- **Progressive Warnings**: Escalating alerts for repeated violations
- **Visual Indicators**: Color-coded status (🟢 Good, 🔴 Violation)
- **Audio Notifications**: Optional sound alerts for violations
- **Evidence Capture**: Automatic screenshot saving with timestamps
- **Detailed Logging**: Complete violation history in database

###  **Violation Categories**
```
 CRITICAL: Multiple voices, security breaches
 WARNING: Head movements, missing hands  
 AUDIO: Background conversations detected
 DEVICE: Prohibited objects identified
```

##  Technical Architecture

###  **Core Technologies**
```python
# Backend Framework
FastAPI 0.120.4          # Modern async web framework
Uvicorn 0.38.0          # ASGI server with high performance

# Computer Vision & AI  
OpenCV 4.12.0           # Computer vision library
MediaPipe 0.10.14       # Google's ML framework
Ultralytics 8.3.223     # YOLOv8 implementation
NumPy 2.2.6            # Numerical computing

# Audio Processing
Web Audio API           # Browser-native audio analysis
AudioContext           # Real-time frequency analysis

# Database & Storage
SQLite3                # Lightweight embedded database  
Python-multipart       # File upload handling
```

###  **AI Models & Processing**
- **MediaPipe Face Mesh**: 468 facial landmarks for head tracking
- **MediaPipe Hands**: Real-time hand pose estimation
- **YOLOv8n**: 80-class object detection (17+ relevant for cheating)
- **Custom Audio Analysis**: Multi-frequency speech pattern recognition

###  **Architecture Pattern**
```
Frontend (SPA)     ←→    Backend (FastAPI)    ←→    AI Engine
├─ HTML/CSS/JS           ├─ REST API              ├─ OpenCV
├─ Web Audio API         ├─ Authentication        ├─ MediaPipe  
├─ Camera Access         ├─ File Upload           ├─ YOLOv8
└─ Real-time UI          └─ Database ORM          └─ Detection Logic
```

##  Database Management

### **View Database Contents:**
```cmd
python -c "
import sqlite3
conn = sqlite3.connect('proctordb.sqlite')
cursor = conn.cursor()

# Show all tables
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
print('Tables:', cursor.fetchall())

# Show user count
cursor.execute('SELECT COUNT(*) FROM users')
print('Users:', cursor.fetchone()[0])

# Show recent violations
cursor.execute('SELECT * FROM violations ORDER BY timestamp DESC LIMIT 5')
print('Recent violations:', cursor.fetchall())
"
```

### **Database Schema:**
```sql
-- Users table
users (id, name, enrollment, password_hash, created_at)

-- Face registrations  
face_registrations (id, user_id, enrollment, image_path, created_at)

-- Exam sessions
exam_sessions (id, user_id, answers, submitted_at, auto_submitted)

-- Violations log
violations (id, session_id, violation_type, timestamp, screenshot_path)
```

##  Security & Privacy

###  **Security Measures**
- **Password Hashing**: SHA-256 with secure salting
- **CORS Protection**: Cross-origin request filtering  
- **File Validation**: Image format and size verification
- **SQL Injection Prevention**: Parameterized queries only
- **Input Sanitization**: All user inputs validated and cleaned
- **Session Management**: Secure user session handling

### **Privacy Compliance**
- **Local Storage**: All data stored locally, no cloud transmission
- **Consent-based**: Explicit camera/microphone permission requests
- **Data Minimization**: Only essential biometric data collected
- **Right to Delete**: Users can clear their data anytime

##  Roadmap & Future Features

###  **Phase 1 (Current)**
-  Real-time face/hand/object detection
-  Audio monitoring and speech detection  
-  Progressive warning system
-  Complete user flow and authentication

###  **Phase 2 (Planned)**
-  **Face Verification**: Match registered face during exam
-  **Analytics Dashboard**: Detailed violation statistics
-  **Multi-user Sessions**: Concurrent exam monitoring
-  **Mobile App**: Native iOS/Android applications
-  **LMS Integration**: Moodle, Canvas, Blackboard support

###  **Phase 3 (Future)**
-  **Advanced AI**: Behavior pattern analysis
-  **Cloud Deployment**: Scalable infrastructure  
-  **ML Improvements**: Adaptive detection algorithms
-  **Institution Features**: Admin panels and reporting

##  Important Notes

###  **Best Practices**
- **Lighting**: Ensure good, even lighting on face
- **Distance**: Sit 2-3 feet from camera for optimal detection
- **Environment**: Maintain quiet space during exams
- **Browser**: Use Chrome/Firefox for best compatibility
- **Performance**: Close unnecessary applications during exams

###  **Known Limitations**
- Requires stable internet for initial model download
- Performance varies with hardware capabilities  
- May need sensitivity adjustment for different environments
- Works best with standard webcam resolution (720p+)

##  Troubleshooting Guide

###  **Camera Issues**
```bash
# Check camera permissions
# Chrome: Settings > Privacy > Camera
# Firefox: Settings > Privacy > Camera

# Camera not starting:
1. Refresh page and allow permissions
2. Close other apps using camera
3. Restart browser
4. Check camera works in other apps
```

###  **Audio Issues** 
```bash
# Audio monitoring not working:
1. Allow microphone permissions
2. Check microphone works in other apps  
3. Adjust sensitivity slider in app
4. Try different browser

# Hearing your own voice:
- This should not happen with current version
- All video elements are muted by default
- Audio is analyzed but not played back
```

###  **Server Issues**
```bash
# Backend won't start:
python --version          # Check Python 3.8+
pip install --upgrade pip # Update pip
pip install -r requirements.txt  # Reinstall dependencies

# Port conflicts:
netstat -an | findstr :8002  # Check if port in use
# Kill process or change port in backend_server.py

# Database errors:
# Delete proctordb.sqlite and restart (loses data)
# Or check file permissions
```

###  **Browser Issues**
```bash
# Page not loading:
1. Check both servers are running
2. Try http://localhost:3000/ProctorVision.html
3. Clear browser cache and cookies
4. Try incognito/private mode
5. Check browser developer console for errors
```

##  Support & Contact

For issues, feature requests, or contributions:

-  **Bug Reports**: Check console logs and provide error details
-  **Feature Requests**: Describe use case and expected behavior  
-  **Contributions**: Fork repository and submit pull requests
-  **Technical Support**: Include system specs and error messages

---

##  Project Highlights

**ProctorVision** represents a complete, production-ready exam proctoring solution featuring:

 **Advanced AI Integration** - Computer vision + audio analysis  
 **Real-world Ready** - Handles permissions, errors, edge cases  
 **Security Focused** - Privacy-compliant with local data storage  
 **User Experience** - Intuitive interface with clear feedback  
 **Performance Optimized** - Efficient detection with minimal resource usage  
 **Developer Friendly** - Clean code, comprehensive documentation  

**Built for the future of secure, intelligent exam monitoring!** 

---

**ProctorVision AI** - Secure, intelligent exam proctoring for the digital age! 