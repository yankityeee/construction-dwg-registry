import streamlit as st
import os
import gc
import json
import tempfile
import shutil
import zipfile
import easyocr  # Swapped from paddleocr
import fitz      # PyMuPDF
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from google import genai
from google.genai import types

# ==========================================
# CONFIGURATION & MAPPINGS
# ==========================================
BATCH_SIZE = 1
CLASS_NAMES = ['drawings_0', 'drawings_180', 'drawings_270', 'drawings_90', 'non_drawings']

GEMINI_SYSTEM_INSTRUCTION = """
Task: Extract 'Drawing Title' and 'Drawing Number' from messy construction drawing OCR text. Output clean values or "" if missing.

RULES:
- TITLE: Target the primary sheet subject. Combine multi-line fragments. Intelligently reconstruct mangled OCR into standard engineering terms.
- EXCLUDE FROM TITLE: Overarching project names, company/departments names, dates, and internal canvas labels.
- NUMBER: Target the unique alphanumeric sheet ID. Often located at the very end of the text block or next to distorted anchors.
- EXCLUDE FROM NUMBER: Decoy numbers like referenced drawings, dates, scales, or detached revision codes.
"""

GEMINI_CONFIG = types.GenerateContentConfig(
    system_instruction=GEMINI_SYSTEM_INSTRUCTION,
    response_mime_type="application/json",
    response_schema={
        "type": "OBJECT",
        "properties": {
            "drawing_title": {"type": "STRING"},
            "drawing_number": {"type": "STRING"},
        },
        "required": ["drawing_title", "drawing_number"]
    }
)

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

TO_TENSOR = transforms.ToTensor()
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# ==========================================
# CACHED RESOURCE LOADING
# ==========================================
@st.cache_resource
def load_resnet_model(model_path="drawing_classifier.pth"):
    device = torch.device("cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, len(CLASS_NAMES))
    )
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in the repository. Please upload it.")
        st.stop()
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

@st.cache_resource
def load_gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ==========================================
# PROCESSING FUNCTIONS
# ==========================================
def extract_drawing_info(page: fitz.Page, ocr: easyocr.Reader, client: genai.Client) -> dict:
    """
    Extracts text using an in-memory PyMuPDF page object to crop mathematical areas BEFORE rendering them to images.
    """
    rect = page.rect
    w, h = rect.width, rect.height
    
    # Define crop rectangles based on drawing orientation
    crop_rects = []
    is_large = (w * h) > (7016 * 9933) # larger than A1 paper size
    
    if h > w: # Portrait
        y_start = h * 0.8
        if is_large:
            mid_w = w / 2
            crop_rects.extend([
                fitz.Rect(0, y_start, mid_w, h),
                fitz.Rect(mid_w, y_start, w, h)
            ])
        else:
            crop_rects.append(fitz.Rect(0, y_start, w, h))
    else: # Landscape
        y_start = h * 0.8
        x_start = w * 0.8
        if is_large:
            mid_w, mid_h = w / 2, h / 2
            crop_rects.extend([
                fitz.Rect(0, y_start, mid_w, h),
                fitz.Rect(mid_w, y_start, w, h),
                fitz.Rect(x_start, 0, w, mid_h),
                fitz.Rect(x_start, mid_h, w, h)
            ])
        else:
            crop_rects.extend([
                fitz.Rect(0, y_start, w, h),
                fitz.Rect(x_start, 0, w, h)
            ])
            
    raw_text = ""
    zoom = 2.0 # Renders at roughly 144 DPI - balance between OCR quality and RAM
    mat = fitz.Matrix(zoom, zoom)
    
    for clip_rect in crop_rects:
        # Render ONLY the specific cropped area to an image
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        # Convert RGBA to RGB if necessary for EasyOCR
        if pix.n == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        # EasyOCR returns a list of tuples: (bounding_box, text, confidence)
        results = ocr.readtext(img_array)
        text_lines = [res[1] for res in results]
        
        if text_lines:
            raw_text += " " + " ".join(text_lines)
            
        # Aggressive garbage collection for RAM safety
        del pix, img_array, results
        gc.collect()

    if not raw_text.strip():
        return {"drawing_title": "", "drawing_number": ""}

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=raw_text.strip(),
            config=GEMINI_CONFIG
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {"drawing_title": "", "drawing_number": ""}

def preprocess_pdf_page(fitz_page):
    # Render at the strict 100 DPI the drawing classifier model expects
    pix = fitz_page.get_pixmap(dpi=100)
    
    # Handle both RGB and RGBA (transparency) from PyMuPDF
    img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    if pix.n == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    thickened_gray = cv2.erode(gray, kernel, iterations=1)
    final_img = cv2.resize(thickened_gray, (448, 448), interpolation=cv2.INTER_AREA)
    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    
    tensor = TO_TENSOR(final_img_rgb)
    
    # Return BOTH the AI tensor and the 100 DPI base image
    return NORMALIZE(tensor), img_np

def classify_image_batch(batch_tensors: list, model: nn.Module, device: torch.device) -> tuple:
    batch_tensor = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted_indices = torch.max(probabilities, 1)
    pred_classes = [CLASS_NAMES[idx.item()] for idx in predicted_indices]
    conf_percents = [conf.item() * 100 for conf in confidences]
    return pred_classes, conf_percents

def process_and_save_page(fitz_doc, pred_class: str, output_dir: str, base_name: str, current_page_num: int, total_pages: int):
    """Fire-and-forget saving of the PDF page."""
    target_folder_name = FOLDER_MAPPING[pred_class]
    out_folder = os.path.join(output_dir, target_folder_name)
    os.makedirs(out_folder, exist_ok=True)
    
    out_pdf_name = f"{base_name}.pdf" if total_pages == 1 else f"{base_name}_p{current_page_num}.pdf"
    out_pdf_path = os.path.join(out_folder, out_pdf_name)

    new_pdf = fitz.open()
    page_index = current_page_num - 1
    new_pdf.insert_pdf(fitz_doc, from_page=page_index, to_page=page_index)

    # Rotation is already handled on the fitz_doc in memory in the main loop
    new_pdf.save(out_pdf_path, garbage=4, deflate=True)
    new_pdf.close()

def create_zip_file(folder_path, output_path):
    """
    Creates a zip archive from a directory.
    output_path should be the absolute base path (without .zip extension).
    """
    # shutil.make_archive automatically appends '.zip' and returns the full path
    final_zip_path = shutil.make_archive(output_path, 'zip', folder_path)
    return final_zip_path

# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
st.set_page_config(page_title="Drawing Registry App", layout="wide")
st.title("🏗️ Construction Drawing Registry App", anchor=False)
st.markdown("Upload your PDF drawings. The AI will classify them, fix their rotation, and generate drawing list.")

# Initialize session state to handle clearing the file uploader
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.spinner("Loading AI Models... (This takes a moment on startup)"):
    model, device = load_resnet_model()
    ocr = load_ocr()
    client = load_gemini_client()

# 1. Original Uploader Layout
uploaded_files = st.file_uploader(
    "",
    type="pdf", 
    accept_multiple_files=True, 
    key=str(st.session_state.uploader_key)
)

if uploaded_files:
    # Handle Duplicate Filenames
    unique_files = {}
    duplicates = []
    
    for file in uploaded_files:
        if file.name in unique_files:
            duplicates.append(file.name)
        else:
            unique_files[file.name] = file
            
    if duplicates:
        st.warning(f"⚠️ Removed duplicate files with the same name: {', '.join(set(duplicates))}")
        
    files_to_process = list(unique_files.values())

    with st.expander("⚙️ Output Preferences"):    
        SAVE_DRAWINGS_FOLDER = st.toggle("Save 'Drawings' Folder", value=True)
        SAVE_NON_DRAWINGS_FOLDER = st.toggle("Save 'Non-Drawings' Folder", value=True)
        GENERATE_CSV_REPORT = st.toggle("Generate CSV Report", value=True)
        
        INCLUDE_MODEL_OUTPUT = False
        if GENERATE_CSV_REPORT:
            INCLUDE_MODEL_OUTPUT = st.toggle("Include Model Predictions & Confidence in CSV", value=False)

    # 1. Evaluate the State
    no_outputs_selected = not (SAVE_DRAWINGS_FOLDER or SAVE_NON_DRAWINGS_FOLDER or GENERATE_CSV_REPORT)

    st.write("---") # Visual divider before actions
    
    # 2. Provide Immediate Feedback
    if no_outputs_selected:
        st.warning("⚠️ Please select at least one output preference to begin processing.")

    # Buttons Layout and Primary Styling
    col1, col2, col3 = st.columns([2, 2, 4]) 
    
    with col1:
        # 3. Disable the Action Button
        start_processing = st.button(
            "▶️ Start Processing", 
            type="primary", 
            use_container_width=True,
            disabled=no_outputs_selected
        )
    with col2:
        if st.button("🗑️ Clear Uploads", use_container_width=True):
            st.session_state.uploader_key += 1
            st.rerun()

    if start_processing:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "inputs")
            output_dir = os.path.join(temp_dir, "outputs")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            all_results = []
            live_log_data = [] 
            
            # --- UI DASHBOARD PLACEHOLDERS ---
            progress_bar = st.progress(0)
            live_header = st.empty() # Added placeholder for the header
            
            live_header.subheader("Processing View 📸", anchor=False)
            dash_col1, dash_col2 = st.columns([1, 1])
            with dash_col1:
                image_placeholder = st.empty() 
            with dash_col2:
                log_placeholder = st.empty()


            # --- START PROCESSING FILES ---
            # Save only the unique files
            for file in files_to_process:
                file_path = os.path.join(input_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            pdf_files = os.listdir(input_dir)
            total_files = len(pdf_files)

            # Pre-scan to count total pages across all files
            total_pages_all_files = 0
            for filename in pdf_files:
                pdf_path = os.path.join(input_dir, filename)
                try:
                    temp_doc = fitz.open(pdf_path)
                    total_pages_all_files += len(temp_doc)
                    temp_doc.close()
                except:
                    pass # Ignore errors here, catch them during real processing
            
            # Prevent division by zero just in case
            if total_pages_all_files == 0:
                total_pages_all_files = 1 
                
            processed_pages = 0 # Initialize our running counter

            for idx, filename in enumerate(pdf_files):
                pdf_path = os.path.join(input_dir, filename)
                base_name = os.path.splitext(filename)[0]
                
                try:
                    fitz_doc = fitz.open(pdf_path)
                    total_pages = len(fitz_doc)
                    batch_tensors, batch_page_nums = [], []

                    for page_num in range(total_pages):
                        fitz_page = fitz_doc[page_num]
                        
                        # --- RENDER ONCE, USE TWICE ---
                        # Extract both the tensor for the AI and the numpy array for the UI
                        tensor, display_img_np = preprocess_pdf_page(fitz_page)
                        
                        batch_tensors.append(tensor)
                        batch_page_nums.append(page_num + 1)

                        # --- LIVE IMAGE PREVIEW ---
                        # Streamlit reads numpy arrays natively, no conversion needed.
                        # Using container width prevents horizontal layout shifts and flickering.
                        image_placeholder.image(
                            display_img_np, 
                            caption=f"Live View: {filename} (Page {page_num + 1})", 
                            use_container_width=True
                        )
                        # --------------------------

                        if len(batch_tensors) == BATCH_SIZE or page_num == total_pages - 1:
                            pred_classes, conf_percents = classify_image_batch(batch_tensors, model, device)

                            for i, (pred_class, conf_percent) in enumerate(zip(pred_classes, conf_percents)):
                                current_page_num = batch_page_nums[i]
                                target_folder_name = FOLDER_MAPPING[pred_class]
                                degrees_to_fix = ROTATION_FIXES.get(pred_class, 0)
                                
                                # 1. Rotate First: Apply to fitz_doc in-memory
                                current_page = fitz_doc[current_page_num - 1]
                                if degrees_to_fix != 0:
                                    current_page.set_rotation((current_page.rotation + degrees_to_fix) % 360)

                                # (The redundant step 2 UI Preview block has been completely removed)

                                row_data = {
                                    "Folder": target_folder_name,
                                    "Filename": filename,
                                    "Page": current_page_num
                                }

                                # 3. Run OCR In-Memory
                                if target_folder_name == 'drawings':
                                    dwg_info = extract_drawing_info(current_page, ocr, client)
                                    
                                    title = dwg_info.get("drawing_title", "")
                                    number = dwg_info.get("drawing_number", "")
                                    
                                    row_data["Drawing Title"] = title
                                    row_data["Drawing Number"] = number
                                    
                                    if title or number:
                                        live_log_data.append({"Drawing Number": number, "Drawing Title": title})
                                        log_placeholder.dataframe(live_log_data, use_container_width=True)
                                else:
                                    row_data["Drawing Title"] = "N/A"
                                    row_data["Drawing Number"] = "N/A"

                                if INCLUDE_MODEL_OUTPUT:
                                    row_data["Prediction"] = pred_class
                                    row_data["Confidence (%)"] = round(conf_percent, 2)

                                all_results.append(row_data)

                                # 4. Conditionally Save: Fire and forget
                                if (target_folder_name == 'drawings' and SAVE_DRAWINGS_FOLDER) or \
                                   (target_folder_name == 'non_drawings' and SAVE_NON_DRAWINGS_FOLDER):
                                    process_and_save_page(fitz_doc, pred_class, output_dir, base_name, current_page_num, total_pages)

                            batch_tensors, batch_page_nums = [], []
                            gc.collect() # Safe to run, no broken thumbnail references to worry about
                        
                        processed_pages += 1
                        progress_bar.progress(processed_pages / total_pages_all_files)

                    fitz_doc.close()
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")

                progress_bar.progress((idx + 1) / total_files)

            # --- PRE-FINALIZATION CLEANUP ---
            live_header.empty()
            image_placeholder.empty()
            log_placeholder.empty()

            if GENERATE_CSV_REPORT and all_results:
                df = pd.DataFrame(all_results)
                
                cols = ['Folder', 'Filename', 'Page', 'Drawing Title', 'Drawing Number']
                if INCLUDE_MODEL_OUTPUT:
                    cols.extend(['Prediction', 'Confidence (%)'])
                    
                df = df[cols].sort_values(by=['Folder', 'Filename']).reset_index(drop=True)
                
                csv_path = os.path.join(output_dir, 'drawing_registry.csv')
                df.to_csv(csv_path, index=False)
                st.dataframe(df)
            
            # --- FINAL UI CLEANUP ---
            progress_bar.empty() # Remove the progress bar to make the final screen totally clean
            
            st.success("✅ Processing Complete!")
            
            # Step 1: Save the zip strictly inside the temporary directory
            zip_target_path = os.path.join(temp_dir, "processed_drawings")
            zip_path = create_zip_file(output_dir, zip_target_path)
            
            # Step 2: Read the file into memory to release the disk lock before temp_dir closes
            with open(zip_path, "rb") as fp:
                zip_data = fp.read()
                
            st.download_button(
                label="📥 Download Zipped Registry",
                data=zip_data,
                file_name="processed_drawings.zip",
                mime="application/zip",
                type="primary"
            )