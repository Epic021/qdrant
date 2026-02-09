
import os
import sys
from datetime import datetime
from pathlib import Path

# Try to import reportlab, install if not available
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
except ImportError:
    print("Installing reportlab...")
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'reportlab'])
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def create_codebase_report(output_path: str):
    # Setup document
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                           topMargin=0.75*inch, bottomMargin=0.75*inch,
                           leftMargin=0.75*inch, rightMargin=0.75*inch)
    
    # Calculate available width
    available_width = A4[0] - (1.5 * inch)
    
    # Define Styles
    styles = getSampleStyleSheet()
    
    def add_style(name, parent, **kwargs):
        if name in styles:
            s = styles[name]
        else:
            s = ParagraphStyle(name=name, parent=parent)
            styles.add(s)
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    add_style('ReportTitle', styles['Heading1'], fontSize=24, alignment=TA_CENTER, spaceAfter=20)
    add_style('ReportSubtitle', styles['Heading2'], fontSize=16, alignment=TA_CENTER, spaceAfter=20, textColor=colors.darkblue)
    add_style('SectionHeader', styles['Heading2'], fontSize=16, spaceBefore=20, spaceAfter=10, textColor=colors.darkblue, keepWithNext=True)
    add_style('SubSectionHeader', styles['Heading3'], fontSize=14, spaceBefore=12, spaceAfter=6, textColor=colors.black, keepWithNext=True)
    add_style('CustomBody', styles['Normal'], fontSize=10, alignment=TA_JUSTIFY, spaceBefore=6, spaceAfter=6, leading=14)
    # Style for table cells - Left aligned, no indent
    add_style('CellHeader', styles['Normal'], fontSize=10, fontName='Helvetica-Bold', alignment=TA_LEFT, leading=12)
    add_style('CellBody', styles['Normal'], fontSize=10, fontName='Helvetica', alignment=TA_LEFT, leading=12)

    story = []

    # --- Title Page ---
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Rural Healthcare AI", styles['ReportTitle']))
    story.append(Paragraph("Codebase Analysis & Technical Deep Dive", styles['ReportSubtitle']))
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['ReportSubtitle']))
    story.append(PageBreak())

    # --- 1. System Overview ---
    story.append(Paragraph("1. System Overview", styles['SectionHeader']))
    intro = """
    This system is an offline-capable, multi-agent AI pipeline designed for rural healthcare. 
    It ingests multimodal data (audio, images, text, docs), stabilizes it into a canonical format, 
    maintains a longitudinal patient history in a vector database (Qdrant), and safely retrieves 
    similar historical cases to aid decision-making.
    """
    story.append(Paragraph(intro, styles['CustomBody']))

    # --- 2. Solution Pipeline (High Level) ---
    story.append(Paragraph("2. Solution Pipeline (6-Agent Workflow)", styles['SectionHeader']))
    story.append(Paragraph("The system operates as a directed acyclic graph (DAG) managed by LangGraph. Data flows sequentially through six specialized agents:", styles['CustomBody']))
    
    pipeline_data = [
        ["Agent", "Role", "Key Output"],
        ["1. Field Ingestion", "Normalizes raw inputs, transcribes audio, assesses quality.", "Canonical `patient_data` JSON"],
        ["2. Patient Memory", "Generates embeddings, updates Qdrant, summarizes history.", "Updated `patient_current_state`"],
        ["3. Context Builder", "Plans retrieval by defining hard constraints & filters.", "Structured `retrieval_plan`"],
        ["4. Case Retrieval", "Executes hybrid search (Dense + Sparse) with re-ranking.", "List of `retrieved_cases`"],
        ["5. Explanation", "Synthesizes cases into a human-readable explanation (LLM).", "Safety-checked `explanation_text`"],
        ["6. Referral", " deterministic logic to detect care gaps & recurring risks.", "Actionable `followup_nudges`"]
    ]
    
    # Process table data into Paragraphs for wrapping
    table_flowables = []
    for i, row in enumerate(pipeline_data):
        if i == 0:
            style = styles['CellHeader']
        else:
            style = styles['CellBody']
        table_flowables.append([Paragraph(cell, style) for cell in row])

    # Column widths: Total ~6.7 inch. 1.2 + 3.0 + 2.5
    t_pipeline = Table(table_flowables, colWidths=[1.2*inch, 3.0*inch, 2.5*inch])
    t_pipeline.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_pipeline)

    # --- 3. Deep Dive: Component Logic ---
    story.append(Paragraph("3. Deep Dive: Component Logic", styles['SectionHeader']))
    story.append(Paragraph("Detailed analysis of the core files in `my_code/`.", styles['CustomBody']))

    # Helper for deep dive tables
    def create_deep_dive_table(filename, description, core_logic):
        # Header
        story.append(Paragraph(f"File: <code>{filename}</code>", styles['SubSectionHeader']))
        story.append(Paragraph(description, styles['CustomBody']))
        
        # Technical Details Table
        data = [
            [Paragraph("<b>Component</b>", styles['CellHeader']), Paragraph("<b>Implementation Details</b>", styles['CellHeader'])]
        ]
        for key, value in core_logic:
            data.append([
                Paragraph(key, styles['CellBody']), 
                Paragraph(value, styles['CellBody'])
            ])
        
        t = Table(data, colWidths=[1.8*inch, available_width - 1.8*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    # Agent 1 Detail
    create_deep_dive_table(
        "agents.py",
        "Implements the Field Ingestion Agent. Focuses on data stabilization and quality assessment.",
        [
            ("Class Name", "<code>FieldIngestionAgent</code>"),
            ("Processors", "• <code>TextProcessor</code>: Normalizes whitespace, flags sparse text.<br/>• <code>AudioProcessor</code>: Uses OpenAI Whisper for transcription.<br/>• <code>ImageProcessor</code>: Uses Laplacian variance for blur detection.<br/>• <code>DocumentProcessor</code>: Tesseract OCR & PDF extraction."),
            ("Uncertainty Flags", "Calculates boolean flags: <code>text_sparse</code>, <code>audio_noisy</code>, <code>image_unclear</code>, <code>history_partial</code>."),
            ("Output", "Produces a sanitized <code>PatientData</code> object/dict ready for storage.")
        ]
    )

    # Agent 2 Detail
    create_deep_dive_table(
        "patient_memory_agent.py",
        "Manages long-term memory. Handles embedding generation and Qdrant interactions.",
        [
            ("Class Name", "<code>PatientMemoryAgent</code>"),
            ("Embeddings", "Uses <code>MultiModalEmbedder</code> to vectorise text (Sentence-BERT), images (CLIP), and audio (CLAP/Optional)."),
            ("Storage Strategy", "Upserts event payload to Qdrant collection <code>patient_context_events_v1</code>. Payload includes timestamp, program, demographics, and uncertainty flags."),
            ("State Generation", "Retrieves <b>all</b> history for the patient to generate `patient_current_state`, summarizing total visits, recurring themes (via keyword extraction), and aggregate uncertainty.")
        ]
    )

    # Agent 3 Detail
    create_deep_dive_table(
        "context_builder_agent.py",
        "The 'Retrieval Planner'. Prevents unsafe search by defining strict constraints before search happens.",
        [
            ("Class Name", "<code>ContextBuilderAgent</code>"),
            ("Signal Extraction", "Extracts deterministic signals from history: Age Group (Pediatric vs Adult), Pregnancy Status (Pregnant/None), Program Type."),
            ("Constraint Building", "Creates a `retrieval_plan` containing:<br/>• <b>Hard Filters</b>: Program match, Age range (±10y), Pregnancy match.<br/>• <b>Geography</b>: Restricts search to 'same_block' by default.<br/>• <b>Modality Policy</b>: Disables audio/image search if quality flags are raised."),
            ("Safety", "Does NOT perform the search itself. Acts as a safety gate.")
        ]
    )

    # Agent 4 Detail
    create_deep_dive_table(
        "similar_case_retrieval_agent.py",
        "Executes the Hybrid Search logic defined by the Context Builder plan.",
        [
            ("Class Name", "<code>SimilarCaseRetrievalAgent</code>"),
            ("Hybrid Search", "Combines two search methods:<br/>1. <b>Dense</b>: Vector similarity (Cosine/DotProduct) on text embeddings.<br/>2. <b>Sparse</b>: BM25-style keyword matching using `recurring_themes`."),
            ("Re-ranking Logic", "Determinstic formula: <code>Score = (0.6 * Dense) + (0.4 * Sparse) + PayloadBoost</code>.<br/>• Boosts score for matching Village (+0.1) or Program (+0.05).<br/>• Penalizes high uncertainty."),
            ("Filtering", "Applies Qdrant Filters (<code>must</code>/<code>must_not</code>) derived from the retrieval plan. <b>Crucially</b>, it self-excludes the current event ID.")
        ]
    )

    # --- 4. Infrastructure Files ---
    story.append(Paragraph("4. Infrastructure & Utilities", styles['SectionHeader']))
    
    infra_data = [
        ["File", "Description"],
        ["api.py", "FastAPI backend. Defines endpoints (`/process`, `/search`) and constructs the LangGraph workflow graph. Manages state transitions between agents."],
        ["db.py", "SQLite database manager. Handles storage of: Raw files (`uploads/`), Transaction logs, JSON dumps, and final outputs. Ensures full audit trail."],
        ["qdrant_manager.py", "Wrapper for Qdrant Client. Manages connection, collection creation, and search queries."],
        ["visu/generate_*.py", "Utility scripts for generating PDF reports (like this one) using ReportLab."]
    ]
    
    infra_flowables = []
    for i, row in enumerate(infra_data):
        if i == 0:
            style = styles['CellHeader']
        else:
            style = styles['CellBody']
        infra_flowables.append([Paragraph(cell, style) for cell in row])
        
    t_infra = Table(infra_flowables, colWidths=[1.8*inch, available_width - 1.8*inch])
    t_infra.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_infra)

    story.append(Spacer(1, 20))
    story.append(Paragraph("End of Report", styles['ReportSubtitle']))

    doc.build(story)
    print(f"Report generated successfully: {output_path}")

if __name__ == "__main__":
    output_pdf = "visu/Codebase_Analysis_Report.pdf"
    # Ensure visu dir exists
    Path("visu").mkdir(exist_ok=True)
    create_codebase_report(output_pdf)
