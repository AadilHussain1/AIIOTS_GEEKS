@echo off
REM AI Resume Screener - Setup Script for Windows
REM This script automates the installation process

echo ==================================
echo AI Resume Screener - Setup
echo ==================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo Pip upgraded
echo.

REM Install requirements
echo Installing dependencies...
echo This may take several minutes...
pip install -r requirements.txt
echo All dependencies installed
echo.

REM Download BERT model
echo Downloading BERT model (this may take a few minutes)...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
echo BERT model downloaded
echo.

echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo To run the application:
echo 1. Activate the virtual environment:
echo    venv\Scripts\activate
echo.
echo 2. Run the Streamlit app:
echo    streamlit run app.py
echo.
echo 3. Open your browser and go to:
echo    http://localhost:8501
echo.
echo ==================================
pause
