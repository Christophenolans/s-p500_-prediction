@echo off
py -3.13 -m pip install -r requirements.txt
if errorlevel 1 goto fail
py -3.13 main.py
pause
exit /b 0
:fail
echo Installation failed. Try: py -3.13 -m ensurepip --upgrade
echo Then: py -3.13 -m pip install --upgrade pip
pause
