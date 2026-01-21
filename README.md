# Rural Healthcare Memory Assistant

An AI-powered medical case memory assistant designed to help healthcare workers in rural settings by leveraging historical case data and visual morphology analysis.

## 🚀 Overview
Derm-Memory Assistant (Rural Derm-AI) helps clinicians compare current patient visits with a "collective memory" of historical cases. It simulates a patient arrival, displays clinical observations, and provides AI-driven similarity results based on visual and textual data.

## 📁 Project Structure
```text
.
├── LICENSE
├── README.md                   # Project documentation
├── app.py                      # Main Streamlit application
├── data/
│   ├── generate_data.py        # Script to generate synthetic patient cases
│   ├── images/                 # Folder containing patient lesion images
│   └── json/                   # JSON clinical records for cases
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup script
├── src/                        # Placeholder for core backend logic
└── utils/
    └── preprocessing.py        # Data loading and text formatting utilities
```

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher
- `pip` (Python package installer)

### 2. Environment Setup
Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd Convolve-4.0-Submission-2Slow2Serious
```

Create and activate a virtual environment:
```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
```
*Alternatively, install as an editable package:*
```bash
pip install -e .
```

### 4. Data Generation (Optional)
If the `data/` folder is empty or you want to refresh the synthetic data:
```bash
python data/generate_data.py
```

### 5. Run the Application
Start the Streamlit dashboard:
```bash
streamlit run app.py
```

## 🩺 How to Use
1. Use the **Sidebar** to select a "Simulated Patient Arrival".
2. View the **Clinical Observations**, including the captured lesion image and patient history.
3. Click **Search Collective Memory** to find similar historical cases.
4. Review the **AI Memory Results** to assist in clinical decision-making.
