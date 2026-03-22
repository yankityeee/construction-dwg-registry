import os
import gc
import json
import time
import tempfile
import zipfile
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from PIL import Image, ImageDraw
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Bypass the Decompression Bomb pixel limit
Image.MAX_IMAGE_PIXELS = None

# --- Configuration & Mappings ---
BATCH_SIZE = 4 
CLASS_NAMES = ['drawings_0', 'drawings_180', 'drawings_270', 'drawings_90', 'non_drawings']

FOLDER_MAPPING = {
    'drawings_0': 'drawings', 'drawings_90': 'drawings',
    'drawings_180': 'drawings', 'drawings_270': 'drawings',
    'non_drawings': 'non_drawings'
}

ROTATION_FIXES = {
    'drawings_0': 0, 'drawings_90': 270,
    'drawings_180': 180, 'drawings_270': 90,
    'non_drawings': 0
}

# --- PyTorch Transforms ---
TO_TENSOR = transforms.ToTensor()
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class DrawingMetadata(BaseModel):
    drawing_title: str = Field(description="The formal title or name of the drawing. Leave empty if not found.")
    drawing_number: str = Field(description="The specific drawing number or reference code. Leave empty if not found.")

# ==========================================
# CACHED RESOURCES
# ==========================================
@st.cache_resource
def get_llm_client():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        return genai.Client(api_key=api_key)
    except KeyError:
        st.error("🚨 GOOGLE_API_KEY is missing! Please add it in the Streamlit App Settings -> Secrets.")
        return None

@st.cache_resource
def load_pytorch_model():
    model_path = "drawing_classifier.pth" 
    device = torch.device("cpu")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        st.error(f"🚨 Model file '{model_path}' not found in the repository!")
        
    model = model.to(device)
    model.eval()
    return model, device

# ==========================================
# CORE FUNCTIONS
# ==========================================
def preprocess_pdf_page(fitz_page) -> torch.Tensor:
    pix = fitz_page.get_pixmap(dpi=150)
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    thickened_gray = cv2.erode(gray, kernel, iterations=1)
    final_img = cv2.resize(thickened_gray, (448, 448), interpolation=cv2.INTER_AREA)
    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    tensor = TO_TENSOR(final_img_rgb)
    return NORMALIZE(tensor)

def classify_image_batch(batch_tensors: list, model: nn.Module, device: torch.device) -> tuple:
    batch_tensor = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted_indices = torch.max(probabilities, 1)
    pred_classes = [CLASS_NAMES[idx.item()] for idx in predicted_indices]
    conf_percents = [conf.item() * 100 for conf in confidences]
    return pred_classes, conf_percents

def extract_metadata_from_text(raw_text, client):
    if not raw_text or not raw_text.strip() or not client:
        return "", ""
    prompt = f"""
    You are an expert engineering document assistant. Extract the exact Drawing Title and Drawing Number from the raw text below.
    If a value cannot be confidently found, output an empty string "".
    Do not include revision numbers in the Drawing Number unless attached directly to it.

    Raw Text:
    {raw_text}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DrawingMetadata,
                temperature=0.1
            )
        )
        data = json.loads(response.text)
        title = data.get("drawing_title", "").strip()
        number = data.get("drawing_number", "").strip()
        
        if title.lower() in ["not found", "none", "n/a", "null"]: title = ""
        if number.lower() in ["not found", "none", "n/a", "null"]: number = ""
        return title, number
    except Exception:
        return "", ""

def extract_metadata_from_image(fitz_page, client):
    if not client: return "", ""
    
    pix = fitz_page.get_pixmap(dpi=150) 
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Draw a white rectangle over the main drawing area to reduce visual noise
    width, height = img.size
    draw = ImageDraw.Draw(img)
    mask_box = [0, 0, width * 0.80, height * 0.85]
    draw.rectangle(mask_box, fill="white")
    
    prompt = """
    You are an expert engineering document assistant. Look at this engineering drawing.
    Extract the exact Drawing Title and Drawing Number from the title block.
    If a value cannot be confidently found, output an empty string "".
    Do not include revision numbers in the Drawing Number unless attached directly to it.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DrawingMetadata,
                temperature=0.1
            )
        )
        data = json.loads(response.text)
        title = data.get("drawing_title", "").strip()
        number = data.get("drawing_number", "").strip()
        
        if title.lower() in ["not found", "none", "n/a", "null"]: title = ""
        if number.lower() in ["not found", "none", "n/a", "null"]: number = ""
        return title, number
    except Exception as e:
        print(f"Vision extraction failed: {e}")
        return "", ""

def extract_drawing_details(fitz_page, client):
    page_rect = fitz_page.rect
    w, h = page_rect.width, page_rect.height
    
    text_right = fitz_page.get_text("text", clip=fitz.Rect(w * 0.8, 0, w, h))
    text_bottom = fitz_page.get_text("text", clip=fitz.Rect(0, h * 0.85, w * 0.8, h))
    native_text = (text_right + " " + text_bottom).strip()
    
    if len(native_text) > 20:
        words = native_text.split()
        limited_text = " | ".join(words[-100:])
        return extract_metadata_from_text(limited_text, client)
    else:
        return extract_metadata_from_image(fitz_page, client)

def process_single_pdf(pdf_path: str, file_display_name: str, output_dir: str, model: nn.Module, device: torch.device, client, save_dwg, save_non_dwg) -> list:
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    results = []

    try:
        fitz_doc = fitz.open(pdf_path)
        pdf_reader = PdfReader(pdf_path)
        total_pages = len(fitz_doc)

        batch_tensors = []
        batch_page_nums = []
        batch_fitz_pages = []

        for page_num in range(total_pages):
            fitz_page = fitz_doc[page_num]
            batch_tensors.append(preprocess_pdf_page(fitz_page))
            batch_page_nums.append(page_num + 1)
            batch_fitz_pages.append(fitz_page)

            if len(batch_tensors) == BATCH_SIZE or page_num == total_pages - 1:
                pred_classes, conf_percents = classify_image_batch(batch_tensors, model, device)

                for i, (pred_class, conf_percent) in enumerate(zip(pred_classes, conf_percents)):
                    current_page_num = batch_page_nums[i]
                    target_folder_name = FOLDER_MAPPING[pred_class]
                    current_fitz_page = batch_fitz_pages[i]
                    
                    drawing_title = ""
                    drawing_no = ""
                    
                    degrees_to_fix = ROTATION_FIXES.get(pred_class, 0)
                    pdf_page_pypdf = pdf_reader.pages[current_page_num - 1]
                    
                    if degrees_to_fix != 0:
                        pdf_page_pypdf.rotate(degrees_to_fix)
                        current_fitz_page.set_rotation((current_fitz_page.rotation + degrees_to_fix) % 360)
                    
                    if target_folder_name == 'drawings':
                        drawing_title, drawing_no = extract_drawing_details(current_fitz_page, client)

                    if (target_folder_name == 'drawings' and save_dwg) or \
                       (target_folder_name == 'non_drawings' and save_non_dwg):
                        
                        pdf_writer = PdfWriter()
                        pdf_writer.add_page(pdf_page_pypdf)
                        out_folder = os.path.join(output_dir, target_folder_name)
                        os.makedirs(out_folder, exist_ok=True)
                        out_pdf_path = os.path.join(out_folder, f"{base_name}_page_{current_page_num}.pdf")
                        
                        with open(out_pdf_path, "wb") as f_out:
                            pdf_writer.write(f_out)

                    results.append({
                        "Folder": target_folder_name,
                        "Filename": os.path.basename(pdf_path),
                        "Page": current_page_num,
                        "Drawing No.": drawing_no,
                        "Drawing Title": drawing_title,
                        "Prediction": pred_class,
                        "Confidence %": round(conf_percent, 2)
                    })

                batch_tensors, batch_page_nums, batch_fitz_pages = [], [], []
                gc.collect()

        fitz_doc.close()
    except Exception as e:
        st.error(f"Error processing {file_display_name}: {e}")

    return results

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="Drawing Registry App", layout="wide")
st.title("🏗️ Construction Drawing Registry App")
st.write("Upload your combined PDFs. The model will classify pages as drawings or non-drawings, rotate them correctly, and generate drawing list.")

# Load Heavy Resources
model, device = load_pytorch_model()
llm_client = get_llm_client()

# --- UI Session State Management ---
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "default_uploader"

# The file uploader is tied to the dynamic session state key
uploaded_files = st.file_uploader(
    "Upload PDF Files", 
    type=["pdf"], 
    accept_multiple_files=True, 
    key=st.session_state.uploader_key
)

if uploaded_files:
    # --- Modern Toggle UI ---
    st.markdown("**⚙️ Output Preferences**")
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    
    with opt_col1:
        SAVE_DRAWINGS_FOLDER = st.toggle("Drawings Folder", value=True)
    with opt_col2:
        SAVE_NON_DRAWINGS_FOLDER = st.toggle("Non-Drawings Folder", value=True)
    with opt_col3:
        GENERATE_CSV_REPORT = st.toggle("CSV Report", value=True)
        
    st.divider() # Adds a clean horizontal line before the action buttons
    
    # Layout for side-by-side buttons
    col1, col2, col3 = st.columns([2, 2, 6])
    
    with col1:
        start_button = st.button("Start Processing", type="primary")
        
    with col2:
        if st.button("Clear / New Batch", type="secondary"):
            # Change the key to force the uploader to reset, emptying the files
            st.session_state.uploader_key = str(time.time())
            st.rerun()

    # --- Processing Execution ---
    if start_button:
        with tempfile.TemporaryDirectory() as temp_in_dir, tempfile.TemporaryDirectory() as temp_out_dir:
            
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for f in uploaded_files:
                with open(os.path.join(temp_in_dir, f.name), "wb") as f_out:
                    f_out.write(f.getbuffer())
                    
            total_files = len(uploaded_files)
            
            for i, filename in enumerate(os.listdir(temp_in_dir)):
                status_text.text(f"Processing {filename} ({i+1}/{total_files})...")
                pdf_path = os.path.join(temp_in_dir, filename)
                
                file_results = process_single_pdf(
                    pdf_path, filename, temp_out_dir, model, device, llm_client, 
                    SAVE_DRAWINGS_FOLDER, SAVE_NON_DRAWINGS_FOLDER
                )
                all_results.extend(file_results)
                progress_bar.progress((i + 1) / total_files)

            status_text.text("Finalizing output...")

            if GENERATE_CSV_REPORT and all_results:
                df = pd.DataFrame(all_results)
                columns_order = ["Folder", "Filename", "Page", "Drawing No.", "Drawing Title", "Prediction", "Confidence %"]
                df = df[columns_order]
                df = df.sort_values(by=['Folder', 'Filename', 'Page']).reset_index(drop=True)
                csv_path = os.path.join(temp_out_dir, 'drawing_registry.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            zip_path = os.path.join(tempfile.gettempdir(), "processed_registry.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_out_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_out_dir)
                        zipf.write(file_path, arcname)

            status_text.text("✅ Processing Complete!")
            
            # Persist download button directly below the progress indicators
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="📥 Download Registry (ZIP)",
                    data=f,
                    file_name="processed_registry.zip",
                    mime="application/zip",
                    type="primary"
                )