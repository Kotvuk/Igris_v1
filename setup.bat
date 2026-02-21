@echo off
chcp 65001 >nul 2>&1
title IGRIS — Setup

echo.
echo   ██╗ ██████╗ ██████╗ ██╗███████╗
echo   ██║██╔════╝ ██╔══██╗██║██╔════╝
echo   ██║██║  ███╗██████╔╝██║███████╗
echo   ██║██║   ██║██╔══██╗██║╚════██║
echo   ██║╚██████╔╝██║  ██║██║███████║
echo   ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝
echo   Shadow Knight AI Assistant
echo   ─────────────────────────────────
echo.

:: Check Python
echo [1/5] Проверка Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден!
    echo Установите Python 3.10+ с https://python.org
    echo Убедитесь, что отметили "Add Python to PATH"
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo   Python %PYVER% найден ✓

:: Create venv
echo.
echo [2/5] Создание виртуального окружения...
if exist venv (
    echo   venv уже существует, пропускаю.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ОШИБКА: Не удалось создать venv
        pause
        exit /b 1
    )
    echo   venv создан ✓
)

:: Activate and install
echo.
echo [3/5] Установка зависимостей...
call venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo ОШИБКА: Не удалось установить зависимости
    pause
    exit /b 1
)
echo   Зависимости установлены ✓

:: Create directories
echo.
echo [4/5] Создание директорий...
if not exist data mkdir data
if not exist data\gallery mkdir data\gallery
if not exist static mkdir static
echo   Директории созданы ✓

:: Config
echo.
echo [5/5] Проверка конфигурации...
if not exist config.json (
    copy config.example.json config.json >nul
    echo   config.json создан из примера ✓
    echo   ВАЖНО: Добавьте Groq API ключи в config.json!
) else (
    echo   config.json уже существует ✓
)

echo.
echo ══════════════════════════════════════════
echo   IGRIS установлен успешно!
echo.
echo   Для запуска:
echo     1. Добавьте API ключи в config.json
echo     2. Запустите: venv\Scripts\python main.py
echo.
echo   Или просто запустите start.bat
echo ══════════════════════════════════════════
echo.
pause
