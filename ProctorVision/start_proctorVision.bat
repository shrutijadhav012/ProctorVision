@echo off
echo 🚀 Starting ProctorVision Application...
echo.
echo Starting Backend Server (Port 8002)...
start "ProctorVision Backend" cmd /k "cd /d d:\ProctorVision && python backend_server.py"

echo Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

echo Starting Frontend Server (Port 3000)...
start "ProctorVision Frontend" cmd /k "cd /d d:\ProctorVision && python run_frontend.py"

echo.
echo  ProctorVision is starting up!
echo  Frontend: http://localhost:3000/ProctorVision.html
echo  Backend API: http://localhost:8002
echo  API Docs: http://localhost:8002/docs
echo.
echo Press any key to exit this window...
pause >nul