@echo off
echo ========================================
echo   ML PROJECTS - QUICK START
echo ========================================
echo.
echo Choose an option:
echo.
echo 1. Install Dependencies
echo 2. Train First Model (Project 4 - Fastest)
echo 3. Train All Models
echo 4. Test All Models
echo 5. View Project Summary
echo 6. Open Documentation
echo 7. Exit
echo.
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto train_first
if "%choice%"=="3" goto train_all
if "%choice%"=="4" goto test
if "%choice%"=="5" goto summary
if "%choice%"=="6" goto docs
if "%choice%"=="7" goto end

:install
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Done! Press any key to return to menu...
pause >nul
goto start

:train_first
echo.
echo Training Project 4 (Occupancy Detection)...
cd project_04_occupancy_detection
python train.py
cd ..
echo.
echo Done! Press any key to return to menu...
pause >nul
goto start

:train_all
echo.
echo Training all 10 models (this may take 30-60 minutes)...
python run_all_training.py
echo.
echo Done! Press any key to return to menu...
pause >nul
goto start

:test
echo.
echo Testing all models...
python test_all_models.py
echo.
echo Done! Press any key to return to menu...
pause >nul
goto start

:summary
echo.
type PROJECT_SUMMARY.txt
echo.
echo Press any key to return to menu...
pause >nul
goto start

:docs
echo.
echo Opening documentation files...
start QUICK_START.md
start README.md
start ..\MASTER_PLAN.md
echo.
echo Press any key to return to menu...
pause >nul
goto start

:end
echo.
echo Goodbye!
exit
