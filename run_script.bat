@echo off
ECHO Starting Python script...

:: Set the path to the src directory (modify if src is not in the current directory)
SET "SRC_DIR=.\src"

:: Set the Python script name from command-line argument, default to script.py if not provided
SET "SCRIPT_NAME=%1"

:: Optional: Activate virtual environment if it exists
:: Modify VENV_PATH to your virtual environment's activate script if needed
SET "VENV_PATH=.\.venv\Scripts\activate.bat"
IF EXIST "%VENV_PATH%" (
    CALL "%VENV_PATH%"
    ECHO Virtual environment activated.
) ELSE (
    ECHO No virtual environment found, using system Python.
)

:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Python is not installed or not found in PATH.

    @echo off
    PING localhost -n 60 >nul
    :: PAUSE
    EXIT /b 0
)

:: Check if the script exists in the src directory
IF NOT EXIST "%SRC_DIR%\%SCRIPT_NAME%" (
    ECHO Script %SCRIPT_NAME% not found in %SRC_DIR%.

    @echo off
    PING localhost -n 60 >nul
    :: PAUSE
    EXIT /b 0
)

:: Run the Python script
ECHO Running %SCRIPT_NAME%...
ECHO ----------------------------------------
python "%SRC_DIR%\%SCRIPT_NAME%"
ECHO ----------------------------------------

:: Optional: Deactivate virtual environment if activated
IF EXIST "%VENV_PATH%" (
    deactivate
    ECHO Virtual environment deactivated.

    @echo off
    PING localhost -n 60 >nul
    :: PAUSE
    EXIT /b 0
)



