"""
Bulk Upload DS Data to Qdrant
===============================
Uploads all images and JSON data from ds/ folder to Qdrant collection
"""

import pandas as pd
import os
import json
from pathlib import Path
from typing import Dict, List
import time
import uuid
from tqdm import tqdm

from qdrant_manager import PatientMemoryQdrantClient
from embeddings import MultiModalEmbedder

def load_mappings_csv(csv_path: str = "../ds/mappings.csv") -> pd.DataFrame:
    """Load the mappings CSV file"""
    print(f"Loading mappings from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries")
    return df

def process_single_entry(
    row: pd.Series,
    embedder: MultiModalEmbedder,
    qdrant_client: PatientMemoryQdrantClient,
    ds_base_path: str = "../ds"
) -> bool:
    """
    Process a single entry from the CSV and upsert to Qdrant
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Extract data from row
        image_filename = row['file']
        patient_id = row['patient_id']
        caption = row.get('caption', '')
        image_type = row.get('image_type', '[]')
        json_file = row.get('json_file', '')
        
        # Build file paths
        image_path = os.path.join(ds_base_path, 'images', image_filename)
        json_path = os.path.join(ds_base_path, json_file) if json_file else None
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return False
        
        # Load JSON metadata if available
        metadata = {}
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Generate embeddings
        text_for_embedding = f"{caption} {metadata.get('findings', '')}"
        text_embedding = embedder.embed_text(text_for_embedding)
        image_embedding = embedder.embed_image(image_path)
        
        if not text_embedding or not image_embedding:
            print(f"Warning: Could not generate embeddings for {image_filename}")
            return False
        
        # Create event ID
        event_id = str(uuid.uuid4())
        
        # Prepare payload
        payload = {
            "event_id": event_id,
            "patient_hash": patient_id,
            "timestamp": int(time.time()),
            "processed_text": caption,
            "image_path": image_path,
            "image_filename": image_filename,
            "image_type": image_type,
            "caption": caption,
            "metadata": metadata,
            "visit_count": 1,
            "uncertainty": {
                "transcription": False,
                "image_quality": False,
                "missing_history": False
            },
            "data_source": "ds_bulk_upload"
        }
        
        # Upsert to Qdrant
        from qdrant_client.models import PointStruct
        
        point = PointStruct(
            id=event_id,
            vector={
                "text_event": text_embedding,
                "image_event": image_embedding
            },
            payload=payload
        )
        
        qdrant_client.client.upsert(
            collection_name="patient_context_events_v1",
            points=[point]
        )
        
        return True
        
    except Exception as e:
        print(f"Error processing {row.get('file', 'unknown')}: {e}")
        return False

def bulk_upload_ds_data(
    csv_path: str = "../ds/mappings.csv",
    ds_base_path: str = "../ds",
    batch_size: int = 100,
    max_entries: int = None
):
    """
    Bulk upload all DS data to Qdrant
    
    Args:
        csv_path: Path to mappings CSV
        ds_base_path: Base path to ds folder
        batch_size: Number of entries to process before checkpoint
        max_entries: Maximum entries to process (for testing)
    """
    print("=" * 60)
    print("DS Data Bulk Upload to Qdrant")
    print("=" * 60)
    
    # Load CSV
    df = load_mappings_csv(csv_path)
    
    if max_entries:
        df = df.head(max_entries)
        print(f"Limiting to {max_entries} entries for testing")
    
    # Initialize clients
    print("\nInitializing Qdrant and Embedder...")
    qdrant_client = PatientMemoryQdrantClient()
    embedder = MultiModalEmbedder()
    
    # Process entries
    print(f"\nProcessing {len(df)} entries...")
    success_count = 0
    fail_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Uploading"):
        if process_single_entry(row, embedder, qdrant_client, ds_base_path):
            success_count += 1
        else:
            fail_count += 1
        
        # Checkpoint every batch_size entries
        if (idx + 1) % batch_size == 0:
            print(f"\nCheckpoint: {success_count} successful, {fail_count} failed")
    
    print("\n" + "=" * 60)
    print(f"Upload Complete!")
    print(f"Successfully uploaded: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total: {success_count + fail_count}")
    print("=" * 60)

if __name__ == "__main__":
    # Test with small batch first
    # print("Running test upload with 10 entries...")
    # bulk_upload_ds_data(max_entries=10)
    
    # Uncomment to run full upload
    print("\nRunning full upload...")
    bulk_upload_ds_data()
