"""
Orchestration Agent
===================
Component: Workflow Controller

Role:
    - Receive Global Graph State
    - Determine Next Node based on state
    - Handle Errors and Fallbacks
    - Manage State Transitions

Flow:
Field Ingestion -> Context Builder -> Patient Memory -> Retriever -> Recommender -> Reviewer -> END
"""

import logging
from typing import Dict, Any, Optional

# Relative imports for schemas
from ..schema.orchestration_agent import GraphState

# Import Agents (Instructions)
from .field_ingestion_agent import FieldIngestionAgent
from .patient_memory_agent import PatientMemoryAgent
from .context_builder_agent import ContextBuilderAgent
from .retriever_agent import RetrieverAgent
from .recommender_agent import RecommenderAgent
from .reviewer_agent import ReviewerAgent

# Import Schemas for Type Safety
from ..schema.field_ingestion_agent import FieldIngestionInput
from ..schema.context_builder_agent import ContextBuilderInput
from ..schema.patient_memory_agent import PatientMemoryInput
from ..schema.retriever_agent import RetrieverInput
from ..schema.recommender_agent import RecommenderInput
from ..schema.reviewer_agent import ReviewerInput

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("orchestrator")
logger.setLevel(logging.INFO)

# ============================================================================
# ORCHESTRATION AGENT
# ============================================================================

class OrchestrationAgent:
    """
    Orchestration Agent implementation.
    Acts as the graph definition and runner.
    """
    
    def __init__(self):
        # Initialize sub-agents
        self.field_ingestion = FieldIngestionAgent()
        self.context_builder = ContextBuilderAgent()
        self.patient_memory = PatientMemoryAgent()
        self.retriever = RetrieverAgent()
        self.recommender = RecommenderAgent()
        self.reviewer = ReviewerAgent()
    
    def run_field_ingestion(self, state: GraphState) -> GraphState:
        """Run Field Ingestion"""
        state.current_node = "field_ingestion"
        
        input_data = FieldIngestionInput(**state.initial_input)
        output = self.field_ingestion.run(input_data)
        
        state.patient_data = output.patient_data
        state.history_log.extend(output.logs)
        return state

    def run_context_builder(self, state: GraphState) -> GraphState:
        """Run Context Builder"""
        state.current_node = "context_builder"
        
        if not state.patient_data:
             state.error = "Missing PatientData"
             return state

        input_data = ContextBuilderInput(patient_data=state.patient_data)
        output = self.context_builder.run(input_data)
        
        state.retrieval_plan = output.retrieval_plan
        state.context_payload = output.context_payload
        state.history_log.extend(output.logs)
        return state

    def run_patient_memory(self, state: GraphState) -> GraphState:
        """Run Patient Memory"""
        state.current_node = "patient_memory"
        
        if not state.patient_data or not state.context_payload:
            state.error = "Missing Data for Memory"
            return state
            
        input_data = PatientMemoryInput(
            patient_data=state.patient_data, 
            context_payload=state.context_payload
        )
        output = self.patient_memory.run(input_data)
        
        state.patient_current_state = output.patient_current_state
        state.history_log.extend(output.logs)
        return state

    def run_retriever(self, state: GraphState) -> GraphState:
        """Run Retriever"""
        state.current_node = "retriever"
        
        if not state.retrieval_plan or not state.patient_current_state:
            state.error = "Missing inputs for Retriever"
            return state

        input_data = RetrieverInput(
            retrieval_plan=state.retrieval_plan,
            patient_current_state=state.patient_current_state
        )
        output = self.retriever.run(input_data)
        
        state.retrieved_cases = output.retrieved_cases
        state.retrieval_metadata = output.retrieval_metadata
        state.history_log.extend(output.logs)
        return state

    def run_recommender(self, state: GraphState) -> GraphState:
        """Run Recommender"""
        state.current_node = "recommender"
        
        if not state.patient_current_state: # retrieved_cases can be empty
            state.error = "Missing inputs for Recommender"
            return state

        input_data = RecommenderInput(
            patient_current_state=state.patient_current_state,
            retrieved_cases=state.retrieved_cases
        )
        output = self.recommender.run(input_data)
        
        state.case_analysis = output.analysis
        state.history_log.extend(output.logs)
        return state

    def run_reviewer(self, state: GraphState) -> GraphState:
        """Run Reviewer"""
        state.current_node = "reviewer"
        
        if not state.case_analysis:
            state.error = "Missing inputs for Reviewer"
            return state

        input_data = ReviewerInput(
            case_analysis=state.case_analysis,
            retrieved_cases=state.retrieved_cases
        )
        output = self.reviewer.run(input_data)
        
        state.review_verdict = output.verdict
        state.history_log.extend(output.logs)
        return state

    def workflow(self, initial_input: Dict[str, Any]) -> GraphState:
        """Linear execution of the graph (Simulated)"""
        state = GraphState(initial_input=initial_input)
        
        # 1. Field Ingestion
        try:
             state = self.run_field_ingestion(state)
        except Exception as e:
             state.error = f"Ingestion Failed: {str(e)}"
             return state

        # 2. Context Builder
        try:
             state = self.run_context_builder(state)
        except Exception as e:
             state.error = f"Context Builder Failed: {str(e)}"
             return state
             
        # 3. Patient Memory
        try:
             state = self.run_patient_memory(state)
        except Exception as e:
             state.error = f"Patient Memory Failed: {str(e)}"
             return state

        # 4. Retriever
        try:
             state = self.run_retriever(state)
        except Exception as e:
             state.error = f"Retriever Failed: {str(e)}"
             return state
             
        # 5. Recommender
        try:
             state = self.run_recommender(state)
        except Exception as e:
             state.error = f"Recommender Failed: {str(e)}"
             return state
             
        # 6. Reviewer
        try:
             state = self.run_reviewer(state)
        except Exception as e:
             state.error = f"Reviewer Failed: {str(e)}"
             return state
             
        return state