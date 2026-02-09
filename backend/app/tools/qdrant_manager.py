"""
Qdrant Manager Tool
===================
Manages connection to Qdrant cloud and collection operations.
Handles patient context events with multimodal vectors.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    HnswConfigDiff
)

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("qdrant_manager")

class QdrantManager:
    """
    Qdrant client wrapper for Patient Memory Agent.
    """
    
    COLLECTION_NAME = "patient_context_events_v1"
    
    # Vector dimensions
    TEXT_VECTOR_SIZE = 768      # BioBERT
    IMAGE_VECTOR_SIZE = 512     # CLIP
    AUDIO_VECTOR_SIZE = 512     # CLAP
    
    def __init__(self):
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.url = os.getenv("QDRANT_URL")
        
        if not self.api_key or not self.url:
            logger.warning("QDRANT_API_KEY or QDRANT_URL not set. Qdrant operations will fail.")
        
        try:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=30
            )
            logger.info(f"Connected to Qdrant at {self.url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None
    
    def health_check(self) -> bool:
        if not self.client:
            return False
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def create_collection(self, recreate: bool = False) -> bool:
        if not self.client:
            return False
            
        try:
            collections = self.client.get_collections()
            exists = any(c.name == self.COLLECTION_NAME for c in collections.collections)
            
            if exists:
                if recreate:
                    logger.info(f"Deleting existing collection: {self.COLLECTION_NAME}")
                    self.client.delete_collection(self.COLLECTION_NAME)
                else:
                    logger.info(f"Collection '{self.COLLECTION_NAME}' already exists")
                    return True
            
            logger.info(f"Creating collection: {self.COLLECTION_NAME}")
            
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config={
                    "text_event": VectorParams(size=self.TEXT_VECTOR_SIZE, distance=Distance.COSINE),
                    "image_event": VectorParams(size=self.IMAGE_VECTOR_SIZE, distance=Distance.COSINE),
                    "audio_event": VectorParams(size=self.AUDIO_VECTOR_SIZE, distance=Distance.COSINE)
                },
                hnsw_config=HnswConfigDiff(m=32, ef_construct=200, on_disk=True)
            )
            
            self._create_payload_indexes()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
            
    def _create_payload_indexes(self):
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
            except Exception as e:
                logger.warning(f"Could not index {field_name}: {e}")

    def upsert_event(
        self,
        event_id: str,
        payload: Dict[str, Any],
        text_vector: Optional[List[float]] = None,
        image_vector: Optional[List[float]] = None,
        audio_vector: Optional[List[float]] = None
    ) -> bool:
        if not self.client:
            return False
            
        try:
            vectors = {}
            if text_vector: vectors["text_event"] = text_vector
            if image_vector: vectors["image_event"] = image_vector
            if audio_vector: vectors["audio_event"] = audio_vector
            
            if not vectors:
                logger.warning(f"No vectors provided for event {event_id}")
                return False
            
            point = PointStruct(
                id=event_id,
                vector=vectors,
                payload=payload
            )
            
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[point]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert event {event_id}: {e}")
            return False

    def get_patient_history(self, patient_hash: str, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.client:
            return []
            
        try:
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
                with_vectors=False
            )
            
            points = results[0]
            sorted_points = sorted(points, key=lambda p: p.payload.get("timestamp", 0))
            return [point.payload for point in sorted_points]
            
        except Exception as e:
            logger.error(f"Failed to retrieve history for {patient_hash}: {e}")
            return []

    def search_similar(
        self,
        query_vector: Union[List[float], tuple],
        query_filter: Optional[Any] = None,
        limit: int = 10
    ) -> List[Any]:
        if not self.client:
            return []
            
        try:
            # Determine vector name for named vectors
            vector_name = "text_event"  # Default to text
            vector = query_vector
            
            if isinstance(query_vector, tuple):
                vector_name, vector = query_vector
                
            # Use query_points (newer qdrant-client API)
            if hasattr(self.client, "query_points"):
                result = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=vector,
                    using=vector_name,
                    limit=limit,
                    with_payload=True
                )
                return result.points if hasattr(result, 'points') else []
            # Fallback to search (older API)
            elif hasattr(self.client, "search"):
                return self.client.search(
                    collection_name=self.COLLECTION_NAME,
                    query_vector=(vector_name, vector),
                    limit=limit,
                    with_payload=True
                )
            
            return []
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

