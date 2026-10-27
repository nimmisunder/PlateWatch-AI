from __future__ import annotations

import re
import sqlite3
from datetime import datetime, date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
import os

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "vehicle_entries.db"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="PlateWatch | Vehicle Entry", page_icon="🚘", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f7fb; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px; }
.hero { padding: 1.55rem 1.8rem; border-radius: 22px; background: linear-gradient(120deg,#101d35 0%,#173d60 58%,#176b78 100%); color:white; margin-bottom:1.3rem; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:2rem; margin:0 0 .35rem; }
.hero p { color:#d8e7f2; margin:0; font-size:.98rem; }
.section-title { font-family:'Manrope',sans-serif; color:#172b46; font-size:1.12rem; font-weight:800; margin:.3rem 0 .8rem; }
.plate { display:inline-block; padding:.65rem 1.05rem; border:3px solid #1e293b; border-radius:9px; background:#fff; color:#111827; font-family:monospace; font-size:1.35rem; font-weight:800; letter-spacing:.12em; }
div[data-testid="stMetric"] { background:white; border:1px solid #e5ebf2; padding:14px 18px; border-radius:16px; }
.stButton button { border-radius:10px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS vehicle_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            confidence_note TEXT,
            image_path TEXT,
            status TEXT NOT NULL DEFAULT 'Entry',
            created_at TEXT NOT NULL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_number ON vehicle_entries(plate_number)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entry_time ON vehicle_entries(entry_time)")

init_db()

def load_entries():
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT id, plate_number, entry_time, status, confidence_note, image_path "
            "FROM vehicle_entries ORDER BY entry_time DESC", conn)
    if not df.empty:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    return df

def normalize_plate(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper().strip())

def save_entry(plate, image_path, note="User-confirmed plate; OCR-assisted"):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with get_connection() as conn:
        conn.execute("""INSERT INTO vehicle_entries
            (plate_number, entry_time, confidence_note, image_path, status, created_at)
            VALUES (?, ?, ?, ?, 'Entry', ?)""", (plate, now, note, image_path, now))

def detect_plate(image_rgb):
    """Find a likely plate crop using contour geometry and a simple candidate score."""
    annotated = image_rgb.copy()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 40, 180)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape[:2]
    image_area = float(height * width)
    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / float(h) if h else 0

        # Reject tiny regions and regions that are too large to be a plate.
        if area < image_area * 0.002 or area > image_area * 0.45:
            continue
        if not (2.2 <= ratio <= 6.5 and w >= 80 and h >= 20):
            continue

        # Prefer plate-like proportions and reasonably large regions, without
        # automatically choosing the largest contour (which may include a bumper).
        ratio_score = 1.0 - min(abs(ratio - 4.0) / 4.0, 1.0)
        area_score = min(area / (image_area * 0.12), 1.0)
        score = 0.65 * ratio_score + 0.35 * area_score
        candidates.append((score, x, y, w, h))

    if candidates:
        _, x, y, w, h = max(candidates, key=lambda item: item[0])
        px = max(4, int(w * 0.025))
        py = max(3, int(h * 0.08))
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(width, x + w + px), min(height, y + h + py)
        crop = image_rgb[y1:y2, x1:x2]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (24, 190, 140), 3)
        return crop, annotated, "Likely plate region detected; verify the crop"

    return None, annotated, "No clear plate-shaped region found; OCR will use the full image"


def preprocess_variants(crop):
    """Create several OCR-friendly versions because one thresholding method won't fit every image."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop.copy()
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return [gray, otsu, adaptive]


def run_ocr(image_rgb, use_full_image=False):
    crop, annotated, note = detect_plate(image_rgb)
    if crop is None or use_full_image:
        crop = image_rgb
        if use_full_image:
            note = "OCR attempted on the full image"

    # Run OCR on several preprocessing variants and page segmentation modes.
    # This helps with lighting and crop differences, but cannot remove watermarks.
    configs = [
        "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]

    results = []
    for prepared in preprocess_variants(crop):
        for config in configs:
            raw = pytesseract.image_to_string(prepared, lang="eng", config=config)
            plate = normalize_plate(raw)
            if plate:
                # Prefer common plate-like lengths, then longer text, while still
                # allowing non-US formats. User must confirm before saving.
                length_score = 1.0 if 5 <= len(plate) <= 8 else 0.4
                score = length_score + min(len(plate), 10) * 0.03
                results.append((score, plate))

    plate = max(results, key=lambda item: item[0])[1] if results else ""
    return plate, crop, annotated, note


def pil_to_rgb(image):
    return np.array(ImageOps.exif_transpose(image).convert("RGB"))

with st.sidebar:
    st.markdown("## 🚘 PlateWatch")
    st.caption("Vehicle entry recording")
    page = st.radio("Navigation", ["Overview", "Register vehicle", "Entry records", "About project"], label_visibility="collapsed")
    st.divider()
    st.markdown("**System status**")
    st.success("Local database ready")
    st.caption("OCR may be imperfect. Verify each plate before saving.")

entries = load_entries()
today_count = int((entries["entry_time"].dt.date == date.today()).sum()) if not entries.empty else 0
unique_plates = int(entries["plate_number"].nunique()) if not entries.empty else 0

if page == "Overview":
    st.markdown('<div class="hero"><h1>Vehicle Entry Intelligence</h1><p>Capture a vehicle image, read its registration plate, and keep searchable entry records.</p></div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric("Total entries", f"{len(entries):,}")
    b.metric("Entries today", f"{today_count:,}")
    c.metric("Unique plates", f"{unique_plates:,}")
    left,right = st.columns([1.5,1])
    with left:
        st.markdown('<div class="section-title">Recent activity</div>', unsafe_allow_html=True)
        if entries.empty:
            st.info("No vehicle records yet. Open **Register vehicle** to process your first image.")
        else:
            recent = entries.head(8).copy()
            recent["entry_time"] = recent["entry_time"].dt.strftime("%d %b %Y, %I:%M %p")
            recent = recent.rename(columns={"id":"ID","plate_number":"Plate number","entry_time":"Entry time","status":"Event"})
            st.dataframe(recent[["ID","Plate number","Entry time","Event"]], use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="section-title">Entry trend</div>', unsafe_allow_html=True)
        if entries.empty:
            st.caption("A daily trend will appear once records are saved.")
        else:
            trend = entries.copy()
            trend["Day"] = trend["entry_time"].dt.date
            st.bar_chart(trend.groupby("Day").size().rename("Entries").tail(14), use_container_width=True)
    st.caption("Educational prototype; not a certified ANPR enforcement system.")

elif page == "Register vehicle":
    st.markdown('<div class="hero"><h1>Register a vehicle</h1><p>Upload a clear front or rear vehicle image. Review the OCR output before saving.</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a vehicle image", type=["jpg","jpeg","png","webp"], help="Use a sharp image with the registration plate visible.")
    if uploaded:
        try:
            image_rgb = pil_to_rgb(Image.open(uploaded))
            col1,col2 = st.columns(2, gap="large")
            with col1:
                st.markdown('<div class="section-title">Original image</div>', unsafe_allow_html=True)
                st.image(image_rgb, use_container_width=True)
            with st.form("ocr_form"):
                full_image_mode = st.checkbox("Try OCR on the full image instead of a detected crop")
                submitted = st.form_submit_button("🔎 Detect number plate", type="primary", use_container_width=True)
            if submitted:
                try:
                    with st.spinner("Detecting plate and running OCR..."):
                        plate, crop, annotated, note = run_ocr(image_rgb, full_image_mode)
                    st.session_state["plate_result"] = plate
                    st.session_state["plate_crop"] = crop
                    st.session_state["annotated_image"] = annotated
                    st.session_state["ocr_note"] = note
                except pytesseract.TesseractNotFoundError:
                    st.error("Tesseract OCR was not found. Install it and follow the README setup instructions.")
                except Exception as exc:
                    st.error(f"Image processing failed: {exc}")
            if "plate_result" in st.session_state:
                with col2:
                    st.markdown('<div class="section-title">Detection result</div>', unsafe_allow_html=True)
                    st.image(st.session_state["annotated_image"], caption=st.session_state["ocr_note"], use_container_width=True)
                    st.image(st.session_state["plate_crop"], caption="Region sent to OCR", use_container_width=True)
                result = st.session_state.get("plate_result", "")
                st.markdown("#### OCR result")
                if result:
                    st.markdown(f'<div class="plate">{result}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No readable plate text detected. Try a sharper image or use full-image OCR.")
                confirmed = st.text_input("Confirm or correct the plate number", value=result, max_chars=20,
                                          help="Verify the result against the original image.")
                if st.button("💾 Save vehicle entry", type="primary", disabled=not normalize_plate(confirmed), use_container_width=True):
                    clean = normalize_plate(confirmed)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_path = UPLOAD_DIR / f"{stamp}_{Path(uploaded.name).stem[:40]}.jpg"
                    Image.fromarray(image_rgb).save(image_path, format="JPEG", quality=92)
                    save_entry(clean, str(image_path.relative_to(APP_DIR)))
                    st.success(f"Vehicle entry saved for {clean}.")
                    for key in ("plate_result","plate_crop","annotated_image","ocr_note"):
                        st.session_state.pop(key, None)
                    st.rerun()
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")
    else:
        st.info("Upload an image to begin. For best results, use a straight-on plate photo with good lighting.")

elif page == "Entry records":
    st.markdown('<div class="hero"><h1>Vehicle entry records</h1><p>Search, filter, review, and export recorded vehicle entries.</p></div>', unsafe_allow_html=True)
    entries = load_entries()
    if entries.empty:
        st.info("There are no saved entries yet.")
    else:
        f1,f2,f3 = st.columns([1.4,1,1])
        query = f1.text_input("Search plate number", placeholder="e.g. KL01AB1234")
        date_from = f2.date_input("From date", value=entries["entry_time"].min().date())
        date_to = f3.date_input("To date", value=date.today())
        filtered = entries.copy()
        if query.strip():
            filtered = filtered[filtered["plate_number"].str.contains(normalize_plate(query), case=False, na=False)]
        filtered = filtered[(filtered["entry_time"].dt.date >= date_from) & (filtered["entry_time"].dt.date <= date_to)]
        display = filtered.copy()
        display["entry_time"] = display["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display = display.rename(columns={"id":"ID","plate_number":"Plate number","entry_time":"Entry time","status":"Event","confidence_note":"Notes","image_path":"Image path"})
        st.caption(f"{len(filtered)} record(s) match your filters.")
        st.dataframe(display[["ID","Plate number","Entry time","Event","Notes"]], use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export filtered records as CSV", data=display.drop(columns=["Image path"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
                           file_name="vehicle_entries.csv", mime="text/csv")
        with st.expander("Record management"):
            st.warning("Deleting records is permanent. Export a backup first if needed.")
            delete_id = st.number_input("Record ID to delete", min_value=1, step=1)
            if st.button("Delete selected record"):
                with get_connection() as conn:
                    cursor = conn.execute("DELETE FROM vehicle_entries WHERE id = ?", (int(delete_id),))
                if cursor.rowcount:
                    st.success(f"Record {int(delete_id)} deleted.")
                    st.rerun()
                else:
                    st.error("No record found with that ID.")

elif page == "About project":
    st.markdown('<div class="hero"><h1>About PlateWatch</h1><p>A portfolio-ready prototype for vehicle entry logging using computer vision and OCR.</p></div>', unsafe_allow_html=True)
    st.markdown("### Technology stack")
    st.write("- **Python** — application logic")
    st.write("- **Streamlit** — interactive dashboard")
    st.write("- **OpenCV** — image processing and candidate plate-region detection")
    st.write("- **Tesseract OCR** — optical character recognition")
    st.write("- **SQLite** — persistent local entry records")
    st.write("- **Pandas** — searchable records and CSV export")
    st.markdown("### Current capabilities")
    st.write("- Upload JPG, PNG, and WEBP images")
    st.write("- Detect likely plate-shaped regions and preprocess for OCR")
    st.write("- Manually confirm/correct OCR text before saving")
    st.write("- Store timestamped entries in a local SQLite database")
    st.write("- Dashboard metrics, recent activity, search, date filters, and CSV export")
    st.markdown("### Important limitations")
    st.warning("This is an educational prototype. Detection and OCR can fail on angled, blurred, dirty, low-resolution, or non-standard plates. It does not validate numbers against official records, detect vehicles from live CCTV, or identify vehicle owners. Protect uploaded images and plate data, and obtain appropriate authorization before real-world use.")
