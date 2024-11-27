@echo off

for /F "tokens=1,2 delims==" %%A in (.env) do set %%A=%%B

if not exist "%VENV_NAME%\Scripts\activate" (
    echo Creating New Environment...
    python -m venv %VENV_NAME%
) else (
    echo Using Existing Environment...
)

call %VENV_NAME%\Scripts\activate

if exist %REQUIREMENTS_FILE% (
    echo Installing dependencies...
    pip install -r %REQUIREMENTS_FILE%
) else (
    echo %REQUIREMENTS_FILE% not found, skipping dependency installation.
)

if exist %MAIN_SCRIPTS% (
    echo Running Main Program...
    python %MAIN_SCRIPTS%
) else (
    echo %MAIN_SCRIPTS% not found.
)

deactivate
echo Execution completed successfully.
