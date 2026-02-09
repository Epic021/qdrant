"""
Retriever Agent
===============
Component: Search Execution

Role:
    - Receive `RetrievalPlan` + `PatientCurrentState`
    - Translate plan into Qdrant Filters
    - Execute Hybrid Search (Vector + Keyword) using QdrantManager
    - Apply reranking (if applicable)
    - Output `RetrievedCase` list

Flow:
    Plan -> Filter Construction -> Qdrant Search -> Ranking -> Retrieved Cases
"""

import logging
from typing import List, Dict, Any, Optional

# Import Schemas
from ..schema.retriever_agent import (
    RetrieverInput,
    RetrieverOutput,
    RetrievedCase,
    RetrievalMetadata
)
from ..schema.payload import EventPayload

# Import Tools
from ...tools.qdrant_manager import QdrantManager
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# ============================================================================
# LOGGER
# ============================================================================

class AgentLogger:
    """Simple logger for the agent"""
    def __init__(self):
        self.logs: List[str] = []
    
    def info(self, msg: str):
        self.logs.append(f"[INFO] {msg}")
        print(f"[Retriever] {msg}")
        
    def get_logs(self) -> List[str]:
        return self.logs


# ============================================================================
# RETRIEVER AGENT
# ============================================================================

class RetrieverAgent:
    """
    Retriever Agent implementation using Tools.
    """
    
    def __init__(self, qdrant_manager: Optional[QdrantManager] = None):
        self.qdrant = qdrant_manager or QdrantManager()
        self.logger = AgentLogger()
        
    def _build_filters(self, plan) -> Optional[Filter]:
        """
        Translate RetrievalPlan constraints into Qdrant Filter.
        """
        conditions = []
        constraints = plan.retrieval_constraints
        required = constraints.required_filters
        
        # 1. Program Filter
        if required.program:
            conditions.append(
                FieldCondition(
                    key="program",
                    match=MatchValue(value=required.program)
                )
            )
            self.logger.info(f"Filter added: Program = {required.program}")
            
        # 2. Pregnancy Required
        if required.pregnancy_required:
            conditions.append(
                FieldCondition(
                    key="pregnancy_status",
                    match=MatchValue(value="pregnant")
                )
            )
            self.logger.info("Filter added: Pregnancy required")

        if not conditions:
            return None
            
        return Filter(must=conditions)

    def run(self, input_state: RetrieverInput) -> RetrieverOutput:
        """
        Execute the agent logic.
        """
        plan = input_state.retrieval_plan
        current_state = input_state.patient_current_state
        
        self.logger.info(f"Executing search for patient {current_state.patient_hash}")
        
        # 1. Build Filters
        qdrant_filter = self._build_filters(plan)
        
        self.logger.info("Running Hybrid Search...")
        
        # Use a zero vector for filter-based search
        dummy_vector = [0.0] * 768 
        
        results = self.qdrant.search_similar(
            query_vector=dummy_vector,
            query_filter=qdrant_filter,
            limit=10  # Default limit
        )
        
        self.logger.info(f"Found {len(results)} candidates")
        
        # 2. Parse Results
        retrieved_cases = []
        for hit in results:
            payload = hit.payload
            retrieved_cases.append(RetrievedCase(
                case_id=str(hit.id),
                patient_hash=payload.get("patient_hash", "unknown"),
                similarity_score=hit.score,
                summary=payload.get("processed_text_snippet", "No summary"),
                outcome=payload.get("outcome", "unknown"),
                demographics={"age_group": payload.get("age_group")},
                relevance_reason="Filter match"
            ))
            
        # 3. Metadata
        metadata = RetrievalMetadata(
            total_candidates=len(results),
            filtered_count=len(results),
            final_count=len(retrieved_cases),
            search_strategy="filter_based"
        )
        
        return RetrieverOutput(
            retrieved_cases=retrieved_cases,
            retrieval_metadata=metadata,
            logs=self.logger.get_logs()
        )