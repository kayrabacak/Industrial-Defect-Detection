@echo off
echo ---------------------------------------------------
echo 🏭 Industrial Defect Detection - Baslatiliyor...
echo ---------------------------------------------------

call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [HATA] Sanal ortam (venv) bulunamadi veya aktiflesemedi.
    echo Lutfen once kurulumu tamamlayin.
    pause
    exit /b
)

echo [BILGI] Streamlit paneli aciliyor...
streamlit run dashboard.py

pause
