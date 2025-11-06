# app.py
import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2 # Used for mask overlay and BGR/RGB conversion

# --- AI Model Placeholder Imports ---
# NOTE: Uncomment and adjust these imports when you integrate your real model
# import torch 
# import torchvision.transforms as T 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- END Placeholder Imports ---


# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/segmentation_model.pth' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

# --- LOAD AI MODEL (Placeholder Logic) ---
AI_MODEL_LOADED = False
try:
    # ⚠️ Replace this block with your actual model loading code!
    # e.g., model = torch.load(MODEL_PATH, map_location=device); model.eval()
    if os.path.exists(MODEL_PATH):
        print("✅ Placeholder model file detected.")
        AI_MODEL_LOADED = True
    else:
        print(f"⚠️ WARNING: Model file not found at {MODEL_PATH}. Running in dummy mode.")
        
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    AI_MODEL_LOADED = False


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_segmentation_model(image_path, model_name, sensitivity):
    """
    Runs the segmentation model (or dummy logic) and applies the mask overlay.
    Returns a base64 encoded PNG string of the segmented image.
    """
    try:
        # 1. Load Original Image
        img_pil = Image.open(image_path).convert("RGB")
        img_original_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR) # OpenCV BGR format
        H, W, _ = img_original_cv.shape

        # --- DUMMY SEGMENTATION MASK CREATION (If model isn't integrated) ---
        mask = np.zeros((H, W), dtype=np.uint8)

        # Simulate Crop (value 1) in the center-left
        mask[int(H*0.3):int(H*0.7), int(W*0.2):int(W*0.5)] = 1 
        # Simulate Weed (value 2) based on sensitivity (user input)
        if int(sensitivity) > 50:
            mask[int(H*0.1):int(H*0.3), int(W*0.6):int(W*0.8)] = 2 
        # --------------------------------------------------------------------
        
        # 2. Post-Processing: Apply Mask Overlay
        
        # Define colors (B, G, R) and transparency
        CROP_COLOR = (76, 175, 80)     # Green
        WEED_COLOR = (244, 67, 54)     # Red
        alpha = 0.5 

        overlay = img_original_cv.copy()
        
        # Apply Crop mask
        overlay[mask == 1] = cv2.addWeighted(img_original_cv[mask == 1], 1 - alpha, 
                                             np.array(CROP_COLOR, dtype=np.uint8), alpha, 0)
        # Apply Weed mask
        overlay[mask == 2] = cv2.addWeighted(img_original_cv[mask == 2], 1 - alpha, 
                                             np.array(WEED_COLOR, dtype=np.uint8), alpha, 0)

        # 3. Encode Result
        img_segmented_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        img_segmented = Image.fromarray(img_segmented_rgb)

        buffer = io.BytesIO()
        img_segmented.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return encoded_image

    except Exception as e:
        print(f"Segmentation Error: {e}")
        return None

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment_image():
    """API endpoint to handle image upload and segmentation."""
    file = request.files.get('file')
    model_name = request.form.get('model', 'adaptive_unet')
    sensitivity = request.form.get('sensitivity', 70)

    if not file or file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file or file type not allowed'}), 400

    # 1. Save the uploaded file temporarily
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 2. Run the segmentation model
    base64_segmented_image = run_segmentation_model(filepath, model_name, sensitivity)

    # 3. Clean up and return result
    os.remove(filepath)
    
    if base64_segmented_image:
        return jsonify({
            'success': True,
            'segmented_image': base64_segmented_image
        })
    else:
        return jsonify({'error': 'Segmentation failed on the server.'}), 500

# --- DRIVER CODE ---
if __name__ == '__main__':
    # Use environment variable for port on Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)