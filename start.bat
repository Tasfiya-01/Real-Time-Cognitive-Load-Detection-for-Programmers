@"
@echo off
cd /d C:\Users\HP\Desktop\cognitive_load_thesis
call venv\Scripts\activate.bat
start "Backend" cmd /k "python src/backend_v2.py"
timeout /t 3
cd vscode-extension
call code --install-extension cognitive-load-detector-0.0.1.vsix
cd ..
code .
pause
"@ | Set-Content start.bat