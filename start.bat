@echo off
chcp 65001 >nul
echo ========================================
echo    He thong diem danh bang nhan dien khuon mat
echo ========================================
echo.

cd /d "%~dp0"

echo Khoi dong he thong...
echo Truy cap: http://localhost:5000
echo Nhan Ctrl+C de dung
echo.

.venv\Scripts\python.exe run.py

pause
