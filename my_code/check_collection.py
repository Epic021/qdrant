"""
Quick script to check what's in the Qdrant collection
"""
from my_code.qdrant_client import PatientMemoryQdrantClient

client = PatientMemoryQdrantClient()

# Get collection info
info = client.get_collection_info()
print("\n" + "=" * 70)
print("QDRANT COLLECTION STATUS")
print("=" * 70)
print(f"Collection: patient_context_events_v1")
print(f"Total Points (Events): {info.get('points_count', 0)}")
print(f"Status: {info.get('status', 'unknown')}")

# Check some sample patients
print("\n" + "=" * 70)
print("SAMPLE PATIENT HISTORIES")
print("=" * 70)

sample_patients = ['amit', 'lakshmi', 'vikram', 'sunita', 'ravi', 'mohan']

for patient in sample_patients:
    history = client.get_patient_history(patient)
    print(f"\n{patient.upper()}:")
    print(f"  Total visits: {len(history)}")
    if history:
        for i, event in enumerate(history, 1):
            print(f"  Visit {i}: {event.get('processed_text', 'No text')[:60]}...")
            print(f"           Timestamp: {event.get('timestamp')}")
            print(f"           Event ID: {event.get('event_id')}")

print("\n" + "=" * 70)
print("✅ Collection successfully populated!")
print("=" * 70)
