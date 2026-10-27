# PlateWatch — Vehicle Entry Recording System

A polished Streamlit portfolio prototype that detects a likely vehicle number-plate region, reads the plate using Tesseract OCR, lets a user confirm the result, and stores timestamped records in SQLite.

## Features
- Dashboard metrics, recent activity, and entry trend
- Upload JPG, JPEG, PNG, and WEBP images
- OpenCV preprocessing and candidate plate-region detection
- Tesseract OCR with a plate-character whitelist and fallback segmentation mode
- Manual review/correction before saving
- Persistent SQLite database
- Search by plate number, date filtering, and CSV export
- Local image archive for saved entries

## Requirements
- Python 3.11 recommended
- Tesseract OCR installed separately (the Python package alone is not enough)

## Setup on Windows PowerShell
Run these commands from the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Tesseract OCR for Windows from the official UB Mannheim builds:
https://github.com/UB-Mannheim/tesseract/wiki

If `pytesseract` cannot find Tesseract, add its installation folder to Windows PATH (commonly `C:\Program Files\Tesseract-OCR`) and restart PowerShell. Or add this near the imports in `app.py`, adjusting the path if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Verify it:

```powershell
tesseract --version
```

## Run
```powershell
streamlit run app.py
```
Streamlit will print a local URL, usually `http://localhost:8501`.

## Runtime data
The app creates `data/vehicle_entries.db` and `data/uploads/`. These contain vehicle-related data and are excluded from Git by `.gitignore`. Do not commit real vehicle images, database files, or private plate data to a public repository.

## GitHub
After creating a GitHub repository and configuring its HTTPS remote:

```powershell
git add app.py requirements.txt README.md .gitignore
git commit -m "Build PlateWatch vehicle entry application"
git push -u origin main
```

## Limitations and responsible use
This is an educational prototype, not a certified ANPR system. Detection and OCR may be inaccurate under poor lighting, blur, perspective, or non-standard plates. Always verify results manually. The app does not query official vehicle registries or identify vehicle owners. Use images and registration data only where authorized, and secure stored data.
