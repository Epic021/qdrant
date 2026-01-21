"""
Qdrant Client for Patient Memory Agent
========================================
Manages connection to Qdrant cloud and collection operations.

Collection: patient_context_events_v1
- One point = one patient event (one visit)
- Named vectors: text_event, image_event, audio_event
- Indexed payload fields for filtering
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    PayloadSchemaType,
    HnswConfigDiff
)

# Load environment variables
load_dotenv()


class PatientMemoryQdrantClient:
    """
    Qdrant client wrapper for Patient Memory Agent.
    Handles connection, collection setup, and basic operations.
    """
    
    COLLECTION_NAME = "patient_context_events_v1"
    
    # Vector dimensions (must match embedding models)
    TEXT_VECTOR_SIZE = 768      # BioBERT / sentence-transformers (updated to match actual model)
    IMAGE_VECTOR_SIZE = 512     # CLIP
    AUDIO_VECTOR_SIZE = 512     # CLAP
    
    def __init__(self):
        """Initialize Qdrant client with credentials from .env"""
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.url = os.getenv("QDRANT_URL")
        
        if not self.api_key or not self.url:
            raise ValueError(
                "QDRANT_API_KEY and QDRANT_URL must be set in .env file"
            )
        
        # Initialize client
        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=30
        )
        
        print(f"✓ Connected to Qdrant at {self.url}")
    
    def health_check(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            collections = self.client.get_collections()
            print(f"✓ Qdrant health check passed. Collections: {len(collections.collections)}")
            return True
        except Exception as e:
            print(f"✗ Qdrant health check failed: {e}")
            return False
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create the patient_context_events_v1 collection with named vectors.
        
        Args:
            recreate: If True, delete existing collection and recreate
        
        Returns:
            True if collection created or already exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            exists = any(c.name == self.COLLECTION_NAME for c in collections.collections)
            
            if exists:
                if recreate:
                    print(f"⚠ Deleting existing collection: {self.COLLECTION_NAME}")
                    self.client.delete_collection(self.COLLECTION_NAME)
                else:
                    print(f"✓ Collection '{self.COLLECTION_NAME}' already exists")
                    return True
            
            # Create collection with named vectors
            print(f"Creating collection: {self.COLLECTION_NAME}")
            
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={
                    "text_event": VectorParams(
                        size=self.TEXT_VECTOR_SIZE,
                        distance=Distance.COSINE
                    ),
                    "image_event": VectorParams(
                        size=self.IMAGE_VECTOR_SIZE,
                        distance=Distance.COSINE
                    ),
                    "audio_event": VectorParams(
                        size=self.AUDIO_VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                },
                hnsw_config=HnswConfigDiff(
                    m=32,
                    ef_construct=200,
                    on_disk=True
                )
            )
            
            print(f"✓ Collection '{self.COLLECTION_NAME}' created with named vectors")
            
            # Create payload indexes for filtering
            self._create_payload_indexes()
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to create collection: {e}")
            return False
    
    def _create_payload_indexes(self):
        """Create indexes on payload fields for efficient filtering"""
        
        indexes = [
            ("patient_hash", PayloadSchemaType.KEYWORD),
            ("program", PayloadSchemaType.KEYWORD),
            ("pregnancy_status", PayloadSchemaType.KEYWORD),
            ("block", PayloadSchemaType.KEYWORD),
            ("timestamp", PayloadSchemaType.INTEGER),
        ]
        
        for field_name, field_type in indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=field_type
                )
                print(f"  ✓ Indexed payload field: {field_name} ({field_type.value})")
            except Exception as e:
                print(f"  ⚠ Could not index {field_name}: {e}")
    
    def upsert_event(
        self,
        event_id: str,
        patient_hash: str,
        payload: Dict[str, Any],
        text_vector: Optional[List[float]] = None,
        image_vector: Optional[List[float]] = None,
        audio_vector: Optional[List[float]] = None
    ) -> bool:
        """
        Upsert a patient event to Qdrant.
        
        Args:
            event_id: Unique event identifier (UUID)
            patient_hash: Patient identifier
            payload: Event payload (metadata)
            text_vector: Text embedding vector (optional)
            image_vector: Image embedding vector (optional)
            audio_vector: Audio embedding vector (optional)
        
        Returns:
            True if successful
        """
        try:
            # Build vectors dict (only include non-None vectors)
            vectors = {}
            if text_vector is not None:
                vectors["text_event"] = text_vector
            if image_vector is not None:
                vectors["image_event"] = image_vector
            if audio_vector is not None:
                vectors["audio_event"] = audio_vector
            
            if not vectors:
                print(f"⚠ No vectors provided for event {event_id}")
                return False
            
            # Create point
            point = PointStruct(
                id=event_id,
                vector=vectors,
                payload=payload
            )
            
            # Upsert to collection
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point]
            )
            
            print(f"✓ Upserted event {event_id} for patient {patient_hash}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to upsert event {event_id}: {e}")
            return False
    
    def get_patient_history(
        self,
        patient_hash: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all events for a patient by patient_hash.
        
        Args:
            patient_hash: Patient identifier
            limit: Maximum number of events to retrieve
        
        Returns:
            List of event payloads sorted by timestamp
        """
        try:
            # Query by patient_hash using scroll (no vector search)
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="patient_hash",
                            match=MatchValue(value=patient_hash)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False  # Don't need vectors for history
            )
            
            # Extract points and sort by timestamp
            points = results[0]  # First element is the list of points
            
            if not points:
                print(f"ℹ No history found for patient {patient_hash}")
                return []
            
            # Sort by timestamp
            sorted_points = sorted(
                points,
                key=lambda p: p.payload.get("timestamp", 0)
            )
            
            print(f"✓ Retrieved {len(sorted_points)} event(s) for patient {patient_hash}")
            
            # Return payloads
            return [point.payload for point in sorted_points]
            
        except Exception as e:
            print(f"✗ Failed to retrieve history for {patient_hash}: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.COLLECTION_NAME)
            return {
                "name": info.config.params.vectors,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            print(f"✗ Failed to get collection info: {e}")
            return {}


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("QDRANT CLIENT - DEMO")
    print("=" * 70 + "\n")
    
    # Initialize client
    client = PatientMemoryQdrantClient()
    
    # Health check
    if not client.health_check():
        print("✗ Cannot connect to Qdrant. Check .env configuration.")
        exit(1)
    
    # Create collection
    print("\nCreating collection...")
    client.create_collection(recreate=False)
    
    # Get collection info
    print("\nCollection info:")
    info = client.get_collection_info()
    print(f"  Points count: {info.get('points_count', 'N/A')}")
    print(f"  Status: {info.get('status', 'N/A')}")
    
    print("\n✅ Qdrant client ready!")
