"""
Bulk Data Loader for Patient Memory Agent
==========================================
Loads existing patient_data JSON files from data/ folder and upserts to Qdrant.

This is a one-time setup script to seed the Context Memory with historical data.

Usage:
    python -m my_code.bulk_load_data
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import time

from my_code.patient_memory_agent import PatientMemoryAgent


class BulkDataLoader:
    """
    Loads all patient_data JSON files and processes them through Patient Memory Agent.
    """
    
    def __init__(self, data_dir: str = "data/patient_data"):
        """
        Initialize bulk loader.
        
        Args:
            data_dir: Directory containing patient_data JSON files
        """
        self.data_dir = Path(data_dir)
        self.agent = PatientMemoryAgent()
        self.stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "patients": set()
        }
    
    def load_all(self):
        """Load and process all patient_data JSON files"""
        print("=" * 70)
        print("BULK DATA LOADER - Starting")
        print("=" * 70)
        print(f"Data directory: {self.data_dir}")
        
        if not self.data_dir.exists():
            print(f"✗ Data directory not found: {self.data_dir}")
            return
        
        # Find all JSON files
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            print(f"✗ No JSON files found in {self.data_dir}")
            return
        
        print(f"Found {len(json_files)} patient_data JSON file(s)")
        print("=" * 70)
        
        self.stats["total_files"] = len(json_files)
        
        # Process each file
        for i, json_file in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}] Processing: {json_file.name}")
            print("-" * 70)
            
            try:
                # Load patient_data
                with open(json_file, 'r', encoding='utf-8') as f:
                    patient_data = json.load(f)
                
                # Process through agent
                result = self.agent.process(patient_data)
                
                if result.get("context_memory_written"):
                    self.stats["successful"] += 1
                    self.stats["patients"].add(result.get("patient_hash"))
                    print(f"✓ Successfully processed: {json_file.name}")
                else:
                    self.stats["failed"] += 1
                    print(f"✗ Failed to process: {json_file.name}")
                
            except Exception as e:
                self.stats["failed"] += 1
                print(f"✗ Error processing {json_file.name}: {e}")
            
            # Small delay to avoid overwhelming the connection
            time.sleep(0.1)
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print loading summary"""
        print("\n" + "=" * 70)
        print("BULK DATA LOADER - Summary")
        print("=" * 70)
        print(f"Total files:       {self.stats['total_files']}")
        print(f"Successful:        {self.stats['successful']}")
        print(f"Failed:            {self.stats['failed']}")
        print(f"Unique patients:   {len(self.stats['patients'])}")
        print("=" * 70)
        
        if self.stats["patients"]:
            print(f"\nPatients loaded:")
            for patient in sorted(self.stats["patients"]):
                print(f"  - {patient}")
        
        print("\n✅ Bulk data loading complete!")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Get project root (assuming we're in my_code/)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "patient_data"
    
    loader = BulkDataLoader(data_dir=str(data_dir))
    loader.load_all()
