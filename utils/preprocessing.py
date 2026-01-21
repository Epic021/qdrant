import json
from PIL import Image
import os

# Configuration
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
JSON_DIR = os.path.join(DATA_DIR, "json")

def load_image(filename, target_size=(512, 512)):
    """Loads an image from the data folder and resizes it for consistency."""
    path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('RGB')
    if target_size:
        img = img.resize(target_size)
    return img

def load_case_data(case_id):
    """Loads the JSON data for a specific case ID."""
    filename = f"{case_id}.json"
    path = os.path.join(JSON_DIR, filename)
    
    if not os.path.exists(path):
        return None
        
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def format_clinical_text(data):
    """
    Converts the structured JSON into a readable string for the doctor/UI.
    This is also what we will eventually embed for the 'Text Vector'.
    """
    clinical = data.get("clinical_data", {})
    demographics = data.get("demographics", {})
    
    # Format: "45yo Male from Vellore. Complaint: [Text]..."
    text = f"{demographics.get('age')}yo {demographics.get('gender')} from {demographics.get('location')}.\n"
    text += f"Chief Complaint: {clinical.get('chief_complaint')}\n"
    text += f"History: {clinical.get('history')}\n"
    
    # Add lesions details if they exist
    if "lesion_details" in clinical:
        ld = clinical["lesion_details"]
        text += f"Lesion: {ld.get('color')} {ld.get('elevation')} on {ld.get('location')}. "
        text += f"Size: {ld.get('size_mm')}mm. Bleeding: {ld.get('bleeding')}."
        
    return text