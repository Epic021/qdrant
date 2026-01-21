import json
import random
import os
from datetime import datetime, timedelta

# --- CONFIGURATION ---
OUTPUT_DIR = "data/json"
COUNT = 50

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- TEMPLATES ---
LOCATIONS = ["Vellore_Rural", "Arcot_Clinic", "Ranipet_General", "Ambur_Health_Post", "Chittoor_Border_Unit"]

# Disease Profiles (Updated with Objective Morphology)
DISEASES = {
    "Eczema": {
        "images": [f"skin_eczema_{i:02d}.jpg" for i in range(1, 6)],
        "complaints": ["Intense itching", "Dry, cracked skin", "Red rash that weeps fluid"],
        "history": "History of asthma/hay fever. Flares with stress or harsh soaps.",
        "textures": ["Scaly/Dry", "Lichenified (Thick)", "Crusted"],
        "colors": ["Red", "Pink"],
        "locations": ["Inside Elbows", "Behind Knees", "Hands", "Neck"],
        "treatment": ["Topical Steroids", "Emollients", "Antihistamines"],
        "urgency": "Routine",
        # Objective Morphology
        "elevations": ["Patch", "Plaque (Slightly Raised)"],
        "borders": ["Ill-defined", "Diffuse"],
        "bleeding_prob": 0.4 # From scratching
    },
    "Melanoma": {
        "images": [f"skin_melanoma_{i:02d}.jpg" for i in range(1, 6)],
        "complaints": ["Changing mole", "Mole that bleeds/oozes", "Dark spot with irregular border"],
        "history": "Lesion has changed shape rapidly. History of excessive sun exposure.",
        "textures": ["Irregular", "Nodular", "Ulcerated"],
        "colors": ["Black/Brown/Blue", "Variegated", "Dark Brown"],
        "locations": ["Back", "Legs", "Face", "Chest"],
        "treatment": ["Wide Local Excision", "Sentinal Node Biopsy", "Immunotherapy"],
        "urgency": "Critical Referral",
        # Objective Morphology
        "elevations": ["Macule (Flat)", "Nodule"],
        "borders": ["Irregular", "Notched", "Asymmetric"],
        "bleeding_prob": 0.3
    },
    "Psoriasis": {
        "images": [f"skin_psoriasis_{i:02d}.jpg" for i in range(1, 6)],
        "complaints": ["Silver scaly patches", "Itchy plaques", "Cracked skin that bleeds"],
        "history": "Chronic condition. Worse in winter. Joint pain noted (Psoriatic Arthritis).",
        "textures": ["Silvery Scale", "Thick Plaque", "Rough"],
        "colors": ["Salmon Pink", "Silver/White"],
        "locations": ["Scalp", "Elbows", "Knees", "Lower Back"],
        "treatment": ["Vitamin D analogues", "Phototherapy", "Methotrexate"],
        "urgency": "Routine",
        # Objective Morphology
        "elevations": ["Plaque (Raised)"],
        "borders": ["Well-defined", "Sharp"],
        "bleeding_prob": 0.2 # Auspitz sign
    },
    "Seborrheic Keratosis": {
        "images": [f"skin_seborrheic_{i:02d}.jpg" for i in range(1, 6)],
        "complaints": ["Wart-like growth", "Stuck-on brown patch", "Itchy raised spot"],
        "history": "Slowly appearing over years. 'Stuck on' appearance. Benign nature.",
        "textures": ["Waxy", "Velvety", "Verrucous (Wart-like)"],
        "colors": ["Tan", "Brown", "Black"],
        "locations": ["Face", "Chest", "Shoulders", "Back"],
        "treatment": ["Observation", "Cryotherapy (if irritated)", "Shave Excision"],
        "urgency": "Benign/Routine",
        # Objective Morphology
        "elevations": ["Papule", "Stuck-on Plaque"],
        "borders": ["Well-defined"],
        "bleeding_prob": 0.05
    },
    "Basal Cell Carcinoma": {
        "images": [f"skin_basal_{i:02d}.jpg" for i in range(1, 6)],
        "complaints": ["Pearly bump that won't heal", "Sore that bleeds easily", "Shiny pink patch"],
        "history": "Slow growing over months. Bleeds when towel drying. Sun-exposed area.",
        "textures": ["Pearly/Shiny", "Rolled Borders", "Central Ulcer"],
        "colors": ["Pink", "Translucent", "Red"],
        "locations": ["Nose", "Ear", "Forehead", "Cheek"],
        "treatment": ["Mohs Micrographic Surgery", "Excision", "Topical Imiquimod"],
        "urgency": "Urgent Referral",
        # Objective Morphology
        "elevations": ["Nodule", "Ulcer"],
        "borders": ["Rolled", "Smooth"],
        "bleeding_prob": 0.8
    }
}

def generate_case(index):
    # Pick disease (Uniform distribution)
    disease_name = random.choice(list(DISEASES.keys()))
    profile = DISEASES[disease_name]
    
    # Randomize Lesion Details
    size = round(random.uniform(3.0, 15.0), 1)
    if disease_name == "Melanoma": size += 3.0 
    
    # Randomize Objective Signs
    is_bleeding = random.random() < profile["bleeding_prob"]
    
    case = {
        "id": f"case_{index:03d}",
        "patient_id": f"P_{random.randint(1000, 9999)}",
        "timestamp": (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat(),
        "demographics": {
            "age": random.randint(20, 80),
            "gender": random.choice(["Male", "Female"]),
            "location": random.choice(LOCATIONS)
        },
        "clinical_data": {
            "chief_complaint": random.choice(profile["complaints"]),
            "history": profile["history"],
            "lesion_details": {
                "location": random.choice(profile["locations"]),
                "size_mm": size,
                "texture": random.choice(profile["textures"]),
                "color": random.choice(profile["colors"]),
                "elevation": random.choice(profile["elevations"]), # Objective
                "borders": random.choice(profile["borders"]),      # Objective
                "bleeding": is_bleeding                            # Objective
            }
        },
        "diagnosis": disease_name,
        "treatment_plan": profile["treatment"],
        "outcome": profile["urgency"], 
        "files": {
            "image_main": random.choice(profile["images"])
        }
    }
    return case

print(f"Generating {COUNT} synthetic dermatology cases...")
for i in range(1, COUNT + 1):
    case_data = generate_case(i)
    filename = os.path.join(OUTPUT_DIR, f"case_{i:03d}.json")
    with open(filename, 'w') as f:
        json.dump(case_data, f, indent=2)

print(f" Success! {COUNT} Dermatology cases created in {OUTPUT_DIR}")