"""
PDF Medical Report Generator
Generates sample Lab Reports, Referral Slips, and Discharge Summaries
based on disease images in the data/images folder.
"""

import os
import random
from datetime import datetime, timedelta
from pathlib import Path

# Try to import reportlab, install if not available
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
except ImportError:
    print("Installing reportlab...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'reportlab'])
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# Disease information database
DISEASE_INFO = {
    "basal": {
        "full_name": "Basal Cell Carcinoma",
        "icd_code": "C44.91",
        "description": "A type of skin cancer that begins in the basal cells",
        "symptoms": ["Pearly or waxy bump", "Flat, flesh-colored lesion", "Bleeding or scabbing sore"],
        "tests": ["Skin biopsy", "Dermoscopy", "Histopathological examination"],
        "treatment": ["Surgical excision", "Mohs surgery", "Cryotherapy", "Topical medications"],
        "severity": "Moderate",
        "specialist": "Dermatologist / Oncologist",
        "followup": "3 months"
    },
    "eczema": {
        "full_name": "Atopic Dermatitis (Eczema)",
        "icd_code": "L20.9",
        "description": "A chronic inflammatory skin condition causing itchy, red, dry skin",
        "symptoms": ["Dry, scaly skin", "Intense itching", "Red to brownish patches", "Small raised bumps"],
        "tests": ["Clinical examination", "Patch testing", "Skin prick test", "Blood IgE levels"],
        "treatment": ["Topical corticosteroids", "Moisturizers", "Antihistamines", "Phototherapy"],
        "severity": "Mild to Moderate",
        "specialist": "Dermatologist",
        "followup": "4-6 weeks"
    },
    "melanoma": {
        "full_name": "Malignant Melanoma",
        "icd_code": "C43.9",
        "description": "The most serious type of skin cancer, developing in melanocytes",
        "symptoms": ["Asymmetric mole", "Irregular borders", "Color variation", "Diameter > 6mm", "Evolving shape"],
        "tests": ["Skin biopsy", "Sentinel lymph node biopsy", "CT scan", "PET scan", "Blood LDH"],
        "treatment": ["Wide surgical excision", "Immunotherapy", "Targeted therapy", "Radiation therapy"],
        "severity": "High - URGENT",
        "specialist": "Surgical Oncologist / Dermatologist",
        "followup": "1 month"
    },
    "psoriasis": {
        "full_name": "Psoriasis Vulgaris",
        "icd_code": "L40.0",
        "description": "A chronic autoimmune condition causing rapid skin cell buildup",
        "symptoms": ["Red patches with silvery scales", "Dry, cracked skin", "Itching and burning", "Thickened nails"],
        "tests": ["Clinical examination", "Skin biopsy", "Blood tests for inflammation markers"],
        "treatment": ["Topical corticosteroids", "Vitamin D analogues", "Phototherapy", "Biologics"],
        "severity": "Moderate",
        "specialist": "Dermatologist",
        "followup": "6-8 weeks"
    },
    "seborrheic": {
        "full_name": "Seborrheic Dermatitis",
        "icd_code": "L21.9",
        "description": "A common skin condition causing scaly patches and red skin",
        "symptoms": ["Flaky white or yellow scales", "Red skin", "Itching", "Greasy or oily areas"],
        "tests": ["Clinical examination", "Skin scraping for fungal culture", "Biopsy if unclear"],
        "treatment": ["Antifungal shampoos", "Topical antifungals", "Corticosteroid cream", "Calcineurin inhibitors"],
        "severity": "Mild",
        "specialist": "Dermatologist",
        "followup": "4 weeks"
    }
}

# Sample patient names (Indian context)
PATIENT_NAMES = [
    "Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sunita Devi", "Vikram Singh",
    "Anita Gupta", "Suresh Reddy", "Meena Kumari", "Prakash Rao", "Kavita Nair",
    "Mohan Das", "Lakshmi Iyer", "Arun Joshi", "Geeta Verma", "Ramesh Yadav"
]

# Hospital names
HOSPITALS = [
    "Apollo Multispeciality Hospital",
    "Fortis Healthcare Center", 
    "AIIMS Regional Center",
    "City General Hospital",
    "Max Super Specialty Hospital"
]

# Doctor names
DOCTORS = {
    "Dermatologist": ["Dr. Sanjay Mehta", "Dr. Kavitha Krishnan", "Dr. Raj Malhotra"],
    "Oncologist": ["Dr. Pradeep Sharma", "Dr. Anita Desai", "Dr. Vikrant Kapoor"],
    "General": ["Dr. Suresh Patil", "Dr. Meera Iyer", "Dr. Ramesh Gupta"]
}


def generate_patient_id():
    """Generate a random patient ID"""
    return f"PT{random.randint(100000, 999999)}"


def generate_date(days_ago_range=(1, 30)):
    """Generate a random date within the specified range"""
    days_ago = random.randint(*days_ago_range)
    return (datetime.now() - timedelta(days=days_ago)).strftime("%d-%m-%Y")


def get_styles():
    """Get custom paragraph styles"""
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='Hospital',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=6,
        textColor=colors.darkblue
    ))
    
    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading2'],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=12,
        textColor=colors.black,
        borderColor=colors.black,
        borderWidth=1,
        borderPadding=5
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading3'],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.darkblue
    ))
    
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=3,
        spaceAfter=3
    ))
    
    styles.add(ParagraphStyle(
        name='SmallText',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    ))
    
    return styles


def create_lab_report(output_path: str, disease_key: str, image_path: str):
    """Generate a Lab Report PDF"""
    disease = DISEASE_INFO[disease_key]
    patient_name = random.choice(PATIENT_NAMES)
    patient_id = generate_patient_id()
    hospital = random.choice(HOSPITALS)
    doctor = random.choice(DOCTORS["Dermatologist"])
    
    doc = SimpleDocTemplate(output_path, pagesize=A4, 
                           topMargin=0.5*inch, bottomMargin=0.5*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = get_styles()
    story = []
    
    # Header
    story.append(Paragraph(f"<b>{hospital}</b>", styles['Hospital']))
    story.append(Paragraph("Department of Dermatology & Pathology", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.darkblue))
    story.append(Spacer(1, 10))
    
    # Report Title
    story.append(Paragraph("<b>LABORATORY REPORT</b>", styles['ReportTitle']))
    story.append(Spacer(1, 10))
    
    # Patient Information Table
    patient_data = [
        ["Patient Name:", patient_name, "Patient ID:", patient_id],
        ["Age/Gender:", f"{random.randint(25, 65)} / {random.choice(['Male', 'Female'])}", 
         "Report Date:", datetime.now().strftime("%d-%m-%Y")],
        ["Referring Doctor:", doctor, "Sample Date:", generate_date((1, 5))],
        ["Diagnosis:", disease['full_name'], "ICD Code:", disease['icd_code']]
    ]
    
    patient_table = Table(patient_data, colWidths=[1.3*inch, 2*inch, 1.3*inch, 1.8*inch])
    patient_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 15))
    
    # Clinical Image
    story.append(Paragraph("<b>CLINICAL IMAGE</b>", styles['SectionHeader']))
    try:
        img = Image(image_path, width=2.5*inch, height=2*inch)
        story.append(img)
    except:
        story.append(Paragraph("[Clinical image attached separately]", styles['CustomBody']))
    story.append(Spacer(1, 10))
    
    # Test Results
    story.append(Paragraph("<b>TESTS PERFORMED</b>", styles['SectionHeader']))
    for i, test in enumerate(disease['tests'], 1):
        result = random.choice(["Positive", "Consistent with diagnosis", "Findings noted", "Abnormal"])
        story.append(Paragraph(f"{i}. {test}: <b>{result}</b>", styles['CustomBody']))
    story.append(Spacer(1, 10))
    
    # Observations
    story.append(Paragraph("<b>HISTOPATHOLOGY OBSERVATIONS</b>", styles['SectionHeader']))
    story.append(Paragraph(f"Clinical examination reveals characteristics consistent with {disease['full_name']}.", styles['CustomBody']))
    story.append(Paragraph(f"<b>Observed Symptoms:</b> {', '.join(disease['symptoms'])}", styles['CustomBody']))
    story.append(Spacer(1, 10))
    
    # Conclusion
    story.append(Paragraph("<b>CONCLUSION</b>", styles['SectionHeader']))
    story.append(Paragraph(
        f"Based on clinical presentation and laboratory findings, the patient is diagnosed with "
        f"<b>{disease['full_name']} ({disease['icd_code']})</b>. Severity assessment: <b>{disease['severity']}</b>.",
        styles['CustomBody']
    ))
    story.append(Spacer(1, 20))
    
    # Signature
    story.append(HRFlowable(width="40%", thickness=1, color=colors.black))
    story.append(Paragraph("Pathologist's Signature", styles['SmallText']))
    story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%d-%m-%Y %H:%M')}", styles['SmallText']))
    
    doc.build(story)
    return output_path


def create_referral_slip(output_path: str, disease_key: str, image_path: str):
    """Generate a Referral Slip PDF"""
    disease = DISEASE_INFO[disease_key]
    patient_name = random.choice(PATIENT_NAMES)
    patient_id = generate_patient_id()
    from_hospital = random.choice(HOSPITALS[:2])
    to_hospital = random.choice(HOSPITALS[2:])
    referring_doctor = random.choice(DOCTORS["General"])
    specialist_doctor = random.choice(DOCTORS["Dermatologist"] if disease_key != "melanoma" else DOCTORS["Oncologist"])
    
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                           topMargin=0.5*inch, bottomMargin=0.5*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = get_styles()
    story = []
    
    # Header
    story.append(Paragraph(f"<b>{from_hospital}</b>", styles['Hospital']))
    story.append(Paragraph("Primary Health Care Center", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.darkblue))
    story.append(Spacer(1, 10))
    
    # Title
    story.append(Paragraph("<b>PATIENT REFERRAL SLIP</b>", styles['ReportTitle']))
    story.append(Spacer(1, 10))
    
    # Urgency Banner for serious cases
    if disease_key == "melanoma":
        urgency_style = ParagraphStyle('Urgent', parent=styles['Normal'], 
                                        fontSize=12, alignment=TA_CENTER, 
                                        textColor=colors.white, backColor=colors.red)
        story.append(Paragraph("<b>⚠️ URGENT REFERRAL - SUSPECTED MALIGNANCY ⚠️</b>", urgency_style))
        story.append(Spacer(1, 10))
    
    # Referral Details
    ref_data = [
        ["Referral No:", f"REF{random.randint(10000, 99999)}", "Date:", datetime.now().strftime("%d-%m-%Y")],
        ["From:", from_hospital, "To:", to_hospital],
    ]
    ref_table = Table(ref_data, colWidths=[1.2*inch, 2.3*inch, 0.8*inch, 2.1*inch])
    ref_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightyellow),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(ref_table)
    story.append(Spacer(1, 15))
    
    # Patient Details
    story.append(Paragraph("<b>PATIENT INFORMATION</b>", styles['SectionHeader']))
    patient_data = [
        ["Name:", patient_name, "ID:", patient_id],
        ["Age:", f"{random.randint(25, 65)} years", "Gender:", random.choice(["Male", "Female"])],
        ["Contact:", f"+91 {random.randint(7000000000, 9999999999)}", "Address:", f"Village {random.choice(['A', 'B', 'C'])}, Block {random.randint(1, 10)}"]
    ]
    p_table = Table(patient_data, colWidths=[1*inch, 2*inch, 1*inch, 2.4*inch])
    p_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 15))
    
    # Clinical Details
    story.append(Paragraph("<b>REASON FOR REFERRAL</b>", styles['SectionHeader']))
    story.append(Paragraph(f"<b>Provisional Diagnosis:</b> {disease['full_name']} ({disease['icd_code']})", styles['CustomBody']))
    story.append(Paragraph(f"<b>Presenting Complaints:</b> {', '.join(disease['symptoms'][:3])}", styles['CustomBody']))
    story.append(Paragraph(f"<b>Duration:</b> {random.randint(1, 8)} weeks", styles['CustomBody']))
    story.append(Paragraph(f"<b>Severity:</b> {disease['severity']}", styles['CustomBody']))
    story.append(Spacer(1, 10))
    
    # Referral Request
    story.append(Paragraph("<b>SPECIALIST CONSULTATION REQUESTED</b>", styles['SectionHeader']))
    story.append(Paragraph(f"Please evaluate and manage for suspected {disease['full_name']}.", styles['CustomBody']))
    story.append(Paragraph(f"<b>Referred to:</b> {disease['specialist']}", styles['CustomBody']))
    story.append(Paragraph(f"<b>Suggested Tests:</b> {', '.join(disease['tests'])}", styles['CustomBody']))
    story.append(Spacer(1, 15))
    
    # Clinical Image
    story.append(Paragraph("<b>CLINICAL PHOTOGRAPH (Attached)</b>", styles['SectionHeader']))
    try:
        img = Image(image_path, width=2*inch, height=1.5*inch)
        story.append(img)
    except:
        story.append(Paragraph("[See attached clinical photograph]", styles['CustomBody']))
    story.append(Spacer(1, 20))
    
    # Signatures
    sig_data = [
        ["Referring Physician:", "Receiving Physician:"],
        [referring_doctor, "(To be filled)"],
        ["Signature: _______________", "Signature: _______________"],
        [f"Date: {datetime.now().strftime('%d-%m-%Y')}", "Date: _______________"]
    ]
    sig_table = Table(sig_data, colWidths=[3*inch, 3*inch])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(sig_table)
    
    doc.build(story)
    return output_path


def create_discharge_summary(output_path: str, disease_key: str, image_path: str):
    """Generate a Discharge Summary PDF"""
    disease = DISEASE_INFO[disease_key]
    patient_name = random.choice(PATIENT_NAMES)
    patient_id = generate_patient_id()
    hospital = random.choice(HOSPITALS)
    doctor = random.choice(DOCTORS["Dermatologist"])
    admission_date = generate_date((7, 14))
    discharge_date = datetime.now().strftime("%d-%m-%Y")
    
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                           topMargin=0.5*inch, bottomMargin=0.5*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    styles = get_styles()
    story = []
    
    # Header
    story.append(Paragraph(f"<b>{hospital}</b>", styles['Hospital']))
    story.append(Paragraph("Department of Dermatology", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.darkblue))
    story.append(Spacer(1, 10))
    
    # Title
    story.append(Paragraph("<b>DISCHARGE SUMMARY</b>", styles['ReportTitle']))
    story.append(Spacer(1, 10))
    
    # Patient & Admission Details
    admit_data = [
        ["Patient Name:", patient_name, "Patient ID:", patient_id],
        ["Age/Gender:", f"{random.randint(25, 65)} / {random.choice(['Male', 'Female'])}", 
         "Ward/Bed:", f"Derma-{random.randint(1, 5)}/{random.choice(['A', 'B', 'C'])}{random.randint(1, 10)}"],
        ["Date of Admission:", admission_date, "Date of Discharge:", discharge_date],
        ["Attending Physician:", doctor, "Consultant:", random.choice(DOCTORS["Dermatologist"])]
    ]
    
    admit_table = Table(admit_data, colWidths=[1.4*inch, 1.9*inch, 1.4*inch, 1.7*inch])
    admit_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(admit_table)
    story.append(Spacer(1, 15))
    
    # Diagnosis
    story.append(Paragraph("<b>FINAL DIAGNOSIS</b>", styles['SectionHeader']))
    story.append(Paragraph(f"<b>Primary:</b> {disease['full_name']} ({disease['icd_code']})", styles['CustomBody']))
    story.append(Paragraph(f"<b>Secondary:</b> {random.choice(['None', 'Mild secondary infection', 'Associated pruritus'])}", styles['CustomBody']))
    story.append(Spacer(1, 10))
    
    # Clinical Summary
    story.append(Paragraph("<b>CLINICAL SUMMARY</b>", styles['SectionHeader']))
    story.append(Paragraph(
        f"Patient presented with {', '.join(disease['symptoms'][:2]).lower()}. "
        f"On examination, findings were consistent with {disease['full_name']}. "
        f"Investigations including {', '.join(disease['tests'][:2]).lower()} confirmed the diagnosis.",
        styles['CustomBody']
    ))
    story.append(Spacer(1, 10))
    
    # Treatment Given
    story.append(Paragraph("<b>TREATMENT ADMINISTERED DURING HOSPITAL STAY</b>", styles['SectionHeader']))
    for i, treatment in enumerate(disease['treatment'][:3], 1):
        story.append(Paragraph(f"{i}. {treatment}", styles['CustomBody']))
    story.append(Spacer(1, 10))
    
    # Condition at Discharge
    story.append(Paragraph("<b>CONDITION AT DISCHARGE</b>", styles['SectionHeader']))
    story.append(Paragraph(
        f"Patient is clinically stable with {random.choice(['significant improvement', 'moderate improvement', 'improvement noted'])} in symptoms. "
        f"No active complications at the time of discharge.",
        styles['CustomBody']
    ))
    story.append(Spacer(1, 10))
    
    # Discharge Medications
    story.append(Paragraph("<b>DISCHARGE MEDICATIONS</b>", styles['SectionHeader']))
    meds = [
        ("Tab. " + random.choice(["Cetirizine 10mg", "Levocetirizine 5mg"]), "Once daily", "2 weeks"),
        (random.choice(disease['treatment'][:2]), "As directed", "2 weeks"),
        ("Moisturizing lotion", "Apply twice daily", "Continuous")
    ]
    meds_data = [["Medication", "Dosage", "Duration"]] + [[m[0], m[1], m[2]] for m in meds]
    meds_table = Table(meds_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    meds_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(meds_table)
    story.append(Spacer(1, 10))
    
    # Follow-up Instructions
    story.append(Paragraph("<b>FOLLOW-UP INSTRUCTIONS</b>", styles['SectionHeader']))
    story.append(Paragraph(f"• Follow-up appointment: {disease['followup']} from discharge date", styles['CustomBody']))
    story.append(Paragraph("• Continue prescribed medications as directed", styles['CustomBody']))
    story.append(Paragraph("• Maintain skin hygiene and avoid known irritants", styles['CustomBody']))
    story.append(Paragraph("• Report immediately if symptoms worsen or new symptoms develop", styles['CustomBody']))
    story.append(Spacer(1, 15))
    
    # Warning Signs
    if disease_key == "melanoma":
        warning_style = ParagraphStyle('Warning', parent=styles['CustomBody'], 
                                        textColor=colors.red)
        story.append(Paragraph("<b>⚠️ WARNING SIGNS - Seek Immediate Medical Attention:</b>", warning_style))
        story.append(Paragraph("• Any new skin lesions or changes in existing lesions", styles['CustomBody']))
        story.append(Paragraph("• Lymph node swelling or systemic symptoms", styles['CustomBody']))
        story.append(Spacer(1, 10))
    
    # Signature
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 10))
    sig_data = [
        ["Prepared by:", "Approved by:"],
        ["_______________", "_______________"],
        [doctor, "Medical Superintendent"],
        [f"Date: {discharge_date}", f"Date: {discharge_date}"]
    ]
    sig_table = Table(sig_data, colWidths=[3*inch, 3*inch])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(sig_table)
    
    doc.build(story)
    return output_path


def main():
    """Main function to generate all PDF reports"""
    
    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    images_dir = project_dir / "data" / "images"
    output_dir = script_dir / "generated_reports"
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("📄 MEDICAL PDF REPORT GENERATOR")
    print("=" * 70)
    print(f"\nImages directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print("\n❌ No images found in the images directory!")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    # Group images by disease
    disease_images = {}
    for img in image_files:
        # Extract disease name from filename (e.g., skin_basal_01.jpg -> basal)
        parts = img.stem.split('_')
        if len(parts) >= 2:
            disease_key = parts[1]
            if disease_key in DISEASE_INFO:
                if disease_key not in disease_images:
                    disease_images[disease_key] = []
                disease_images[disease_key].append(img)
    
    print(f"\nDiseases detected: {', '.join(disease_images.keys())}")
    
    # Generate reports for each disease
    generated_files = []
    
    for disease_key, images in disease_images.items():
        disease_name = DISEASE_INFO[disease_key]['full_name']
        print(f"\n{'='*50}")
        print(f"📋 Generating reports for: {disease_name}")
        print(f"{'='*50}")
        
        # Use first image for this disease
        sample_image = str(images[0])
        
        # 1. Lab Report
        lab_report_path = output_dir / f"lab_report_{disease_key}.pdf"
        create_lab_report(str(lab_report_path), disease_key, sample_image)
        generated_files.append(lab_report_path)
        print(f"   ✅ Lab Report: {lab_report_path.name}")
        
        # 2. Referral Slip
        referral_path = output_dir / f"referral_slip_{disease_key}.pdf"
        create_referral_slip(str(referral_path), disease_key, sample_image)
        generated_files.append(referral_path)
        print(f"   ✅ Referral Slip: {referral_path.name}")
        
        # 3. Discharge Summary
        discharge_path = output_dir / f"discharge_summary_{disease_key}.pdf"
        create_discharge_summary(str(discharge_path), disease_key, sample_image)
        generated_files.append(discharge_path)
        print(f"   ✅ Discharge Summary: {discharge_path.name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ REPORT GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Total PDFs generated: {len(generated_files)}")
    print(f"📂 Location: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(generated_files):
        print(f"   • {f.name}")
    
    return generated_files


if __name__ == "__main__":
    main()
