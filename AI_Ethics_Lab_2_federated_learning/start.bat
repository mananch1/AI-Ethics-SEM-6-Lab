@echo off
REM ================================
REM Federated Learning Launcher
REM Usage: start.bat <noise_std>
REM Example: start.bat 0.5
REM ================================

if "%1"=="" (
    echo Usage: start.bat ^<noise_std^>
    echo Example: start.bat 0.1
    exit /b 1
)

set NOISE=%1

echo Starting Federated Learning with noise = %NOISE%
echo.

REM --- Start server ---
start "FL Server" cmd /k ^
cd server ^&^& python server.py %NOISE%

REM Give server time to start
timeout /t 2 > nul

REM --- Start clients ---
start "Client 1" cmd /k ^
cd client1 ^&^& python client.py

start "Client 2" cmd /k ^
cd client2 ^&^& python client.py

start "Client 3" cmd /k ^
cd client3 ^&^& python client.py
