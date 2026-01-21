"""
Sort Script - Generate patient_data.json files for each image
Using the existing text JSON files to map images to their data
Output format matches: greeting_6d40589f_patient_data.json
"""

import json
import os
import random
import uuid
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "images"
TEXT_DIR = BASE_DIR / "text"
DOCS_DIR = BASE_DIR / "docs"
OUTPUT_DIR = BASE_DIR / "patient_data_jsons"

# Disease key mapping
DISEASE_KEY_MAP = {
    "Basal Cell Carcinoma": "basal",
    "Eczema": "eczema",
    "Melanoma": "melanoma",
    "Psoriasis": "psoriasis",
    "Seborrheic Keratosis": "seborrheic"
}

# Sample Hindi text descriptions for each disease
HINDI_DESCRIPTIONS = {
    "basal": [
        "Patient ko chehra par ek chamakdar pink bump hai, jo thik nahi ho raha. 3 mahine se hai, khoon nikalta hai kabhi kabhi.",
        "Naak par ek shiny bump hai, dheere dheere badh raha hai. Towel se ponchne par khoon aata hai.",
        "Kaan ke paas ek pearly sa dana hai, heal nahi ho raha. Dhoop mein zyada rehte hain."
    ],
    "eczema": [
        "Patient Ravi, 28 saal male, Rampur village. Skin mein laal daayian, khujli bahut, 2 hafte se. Eczema jaise dikhta hai, photo li hai.",
        "Haath mein bahut khujli, skin dry aur cracked hai. Sabun lagane se aur bura hota hai.",
        "Kohni ke andar laal patches, bahut khujli. Bachpan se asthma bhi hai patient ko."
    ],
    "melanoma": [
        "Til ka rang badal raha hai, pehle se bada ho gaya. Border irregular hai, specialist dikhana chahiye.",
        "Peeth par ek dark spot hai jo rapidly change ho raha. Kabhi kabhi khoon bhi nikalta hai.",
        "Pair par ek mole hai jo badh raha hai, alag alag color dikhta hai usme."
    ],
    "psoriasis": [
        "Scalp par silver color ke patches hain, bahut khujli. Sardi mein zyada problem hoti hai.",
        "Kohni aur ghutno par thick plaques hain, silver scale ke saath. Joint mein bhi dard hai.",
        "Kamar ke neeche salmon pink patches, silvery scale. Chronic condition hai patient ki."
    ],
    "seborrheic": [
        "Chest par brown colored growth hai, wart jaisa dikhta hai. Stuck-on appearance hai.",
        "Face par tan colored raised spot, slowly appear hua years mein. Itchy hai thoda.",
        "Shoulders par multiple waxy bumps hain, benign lagte hain. Cosmetic concern hai patient ko."
    ]
}

# Sample audio transcripts
AUDIO_TRANSCRIPTS = [
    "ki jistaranim ki pede ki aayu anne vrakshou ki apeksa lambi hoti hai usitre aad bhi druga yuhu",
    "patient ki skin condition dekhi gayi aur treatment suggest kiya gaya hai",
    "dermatology consultation ke liye refer karna padega patient ko",
    "skin lesion ka examination kiya gaya photo bhi li gayi hai",
    "symptoms 2 hafte se hain aur gradually worse ho rahe hain"
]


def get_disease_key_from_image(image_name):
    """Extract disease key from image filename like 'skin_basal_01.jpg'"""
    for key in ["basal", "eczema", "melanoma", "psoriasis", "seborrheic"]:
        if key in image_name.lower():
            return key
    return None


def load_text_files_by_image():
    """Load all text JSON files and index them by image_main"""
    text_by_image = {}
    
    for text_file in TEXT_DIR.glob("*.json"):
        with open(text_file, 'r') as f:
            data = json.load(f)
            image_main = data.get("files", {}).get("image_main", "")
            if image_main:
                if image_main not in text_by_image:
                    text_by_image[image_main] = []
                text_by_image[image_main].append(data)
    
    return text_by_image


def get_docs_for_disease(disease_key):
    """Get all doc files for a given disease type"""
    docs = []
    for doc_file in DOCS_DIR.glob("*.pdf"):
        if disease_key in doc_file.name.lower():
            docs.append(doc_file)
    return docs


def generate_patient_hash():
    """Generate a simple patient hash like 'greeting'"""
    names = ["ravi", "amit", "priya", "sunita", "rajesh", "meena", "vikram", "anita", "suresh", "kavita",
             "mohan", "geeta", "prakash", "lakshmi", "arun", "patient", "case", "sample"]
    return random.choice(names)


def generate_patient_data_json(image_path, case_data, disease_key, docs):
    """Generate patient_data.json format from case data"""
    
    patient_hash = generate_patient_hash()
    interaction_id = str(uuid.uuid4())
    short_id = interaction_id.split('-')[0]
    
    # Get image absolute path
    image_abs_path = str(image_path.absolute())
    
    # Build image quality list (just the one image, or could be empty if not applicable)
    image_quality = [{
        "path": image_abs_path,
        "score": round(random.uniform(0.3, 0.7), 3)
    }]
    
    # Build documents list (may be empty or have 1-3 docs)
    documents = []
    if docs:
        # Randomly include 0-3 docs
        num_docs = random.randint(0, min(3, len(docs)))
        selected_docs = random.sample(docs, num_docs) if num_docs > 0 else []
        
        for doc in selected_docs:
            documents.append({
                "path": str(doc.absolute()),
                "ocr_text": f"Medical document for {case_data.get('diagnosis', 'Unknown')}. Patient details and clinical findings.",
                "ocr_confidence": round(random.uniform(90.0, 98.0), 1)
            })
    
    # Build processed text from case data
    age = case_data.get("demographics", {}).get("age", random.randint(20, 70))
    gender = case_data.get("demographics", {}).get("gender", "Male")
    location = case_data.get("demographics", {}).get("location", "Village")
    complaint = case_data.get("clinical_data", {}).get("chief_complaint", "Skin condition")
    diagnosis = case_data.get("diagnosis", "Unknown")
    
    # Use Hindi description template
    hindi_texts = HINDI_DESCRIPTIONS.get(disease_key, HINDI_DESCRIPTIONS["eczema"])
    processed_text = random.choice(hindi_texts)
    
    # Randomly decide which modalities to include
    include_audio = random.choice([True, False])
    include_docs = len(documents) > 0
    
    patient_data = {
        "patient_hash": patient_hash,
        "interaction_id": interaction_id,
        "processed_text": processed_text
    }
    
    # Add audio only sometimes
    if include_audio:
        patient_data["audio_transcript"] = random.choice(AUDIO_TRANSCRIPTS)
        patient_data["audio_language"] = "hi"
        patient_data["audio_confidence"] = round(random.uniform(0.5, 0.9), 3)
    
    # Add image quality
    patient_data["image_quality"] = image_quality
    
    # Add documents only if available
    if include_docs:
        patient_data["documents"] = documents
    
    # Add uncertainty flags
    patient_data["uncertainty_flags"] = {
        "text_sparse": random.choice([True, False]),
        "audio_noisy": random.choice([True, False]) if include_audio else False,
        "image_unclear": random.choice([True, False]),
        "document_unclear": random.choice([True, False]) if include_docs else False,
        "history_partial": random.choice([True, False])
    }
    
    # Add capture metadata
    patient_data["capture_metadata"] = {
        "device_id": "web_device",
        "offline": random.choice([True, False]),
        "timestamp": int(datetime.now().timestamp())
    }
    
    # Generate filename: {patient_hash}_{short_id}_patient_data.json
    filename = f"{patient_hash}_{short_id}_patient_data.json"
    
    return patient_data, filename


def main():
    """Main function to generate patient_data.json files"""
    
    print("="*60)
    print("GENERATING PATIENT DATA JSONs")
    print("One JSON per image in patient_data.json format")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Get all images
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    print(f"\nFound {len(images)} images")
    
    # Load text files indexed by image
    text_by_image = load_text_files_by_image()
    print(f"Loaded text files for {len(text_by_image)} unique images")
    
    generated = 0
    
    for image_path in images:
        image_name = image_path.name
        disease_key = get_disease_key_from_image(image_name)
        
        if not disease_key:
            print(f"  Skipping {image_name} - unknown disease type")
            continue
        
        # Get case data for this image (or create default)
        case_data_list = text_by_image.get(image_name, [])
        if case_data_list:
            case_data = case_data_list[0]  # Use first matching case
        else:
            # Create default case data
            case_data = {
                "demographics": {"age": random.randint(20, 70), "gender": random.choice(["Male", "Female"]), "location": "Village"},
                "clinical_data": {"chief_complaint": "Skin condition"},
                "diagnosis": disease_key.title()
            }
        
        # Get docs for this disease
        docs = get_docs_for_disease(disease_key)
        
        # Generate patient_data.json
        patient_data, filename = generate_patient_data_json(image_path, case_data, disease_key, docs)
        
        # Save to output directory
        output_path = OUTPUT_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(patient_data, f, indent=2)
        
        print(f"  Created: {filename}")
        generated += 1
    
    print("\n" + "="*60)
    print(f"DONE! Generated {generated} patient_data.json files")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
