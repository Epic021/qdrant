"""
LangGraph Orchestrator
======================
Main workflow orchestration using LangGraph with async WebSocket logging.
"""

import logging
import asyncio
from typing import TypedDict, List, Optional, Dict, Any, Callable

# Import Instruction Agents
from ..library.instructions.field_ingestion_agent import FieldIngestionAgent
from ..library.instructions.context_builder_agent import ContextBuilderAgent
from ..library.instructions.patient_memory_agent import PatientMemoryAgent
from ..library.instructions.retriever_agent import RetrieverAgent
from ..library.instructions.recommender_agent import RecommenderAgent
from ..library.instructions.reviewer_agent import ReviewerAgent

# Import Schemas
from ..library.schema.field_ingestion_agent import FieldIngestionInput, FieldIngestionOutput, PatientData
from ..library.schema.context_builder_agent import ContextBuilderInput, ContextBuilderOutput, ContextPayload, RetrievalPlan
from ..library.schema.patient_memory_agent import PatientMemoryInput, PatientMemoryOutput, PatientCurrentState
from ..library.schema.retriever_agent import RetrieverInput, RetrieverOutput, RetrievedCase
from ..library.schema.recommender_agent import RecommenderInput, RecommenderOutput, CaseAnalysis
from ..library.schema.reviewer_agent import ReviewerInput, ReviewerOutput, ReviewVerdict

# Import Tools
from ..tools.qdrant_manager import QdrantManager
from ..database.db_manager import DatabaseManager
from ..tools.llm_client import GeminiClient

logger = logging.getLogger("orchestrator")


class WorkflowState(TypedDict):
    """Shared state for the workflow"""
    raw_text: Optional[str]
    audio_path: Optional[str]
    image_paths: Optional[List[str]]
    document_paths: Optional[List[str]]
    patient_hash: str
    device_id: str
    offline: bool
    patient_data: Optional[Dict[str, Any]]
    context_payload: Optional[Dict[str, Any]]
    retrieval_plan: Optional[Dict[str, Any]]
    patient_current_state: Optional[Dict[str, Any]]
    retrieved_cases: Optional[List[Dict[str, Any]]]
    case_analysis: Optional[Dict[str, Any]]
    review_verdict: Optional[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]
    logs: List[str]
    error: Optional[str]


class AsyncOrchestrator:
    """
    Async orchestrator that broadcasts logs per agent in real-time.
    """
    
    def __init__(self, log_callback: Optional[Callable] = None, step_callback: Optional[Callable] = None, thinking_callback: Optional[Callable] = None):
        self.log_callback = log_callback
        self.step_callback = step_callback
        self.thinking_callback = thinking_callback
        
        # Shared tools
        self.qdrant = QdrantManager()
        self.db = DatabaseManager()
        self.llm = GeminiClient()
        
        # Agents
        self.field_ingestion = FieldIngestionAgent(db_manager=self.db)
        self.context_builder = ContextBuilderAgent()
        self.patient_memory = PatientMemoryAgent(
            qdrant_manager=self.qdrant, 
            db_manager=self.db
        )
        self.retriever = RetrieverAgent(qdrant_manager=self.qdrant)
        self.recommender = RecommenderAgent(llm_client=self.llm)
        self.reviewer = ReviewerAgent(llm_client=self.llm)
        
        logger.info("Async Orchestrator initialized.")

    async def _log(self, message: str, level: str = "info"):
        """Broadcast log to WebSocket"""
        if self.log_callback:
            await self.log_callback(message, level)
        logger.info(message)

    async def _step(self, step: str, status: str = "active"):
        """Broadcast step change to WebSocket"""
        if self.step_callback:
            await self.step_callback(step, status)
    
    async def _think(self, message: str):
        """Broadcast thinking/planning message"""
        if self.thinking_callback:
            await self.thinking_callback(message)

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Run the full workflow with real-time logging per agent."""
        
        try:
            # ===================== PLANNING PHASE =====================
            await self._think("🧠 Analyzing patient input and planning workflow...")
            await self._log("[Orchestrator] Received patient data. Planning agent pipeline...")
            
            # Count inputs
            has_text = bool(state.get("raw_text"))
            has_audio = bool(state.get("audio_path"))
            has_images = bool(state.get("image_paths"))
            has_docs = bool(state.get("document_paths"))
            
            plan_msg = f"📋 Plan: Process "
            inputs = []
            if has_text: inputs.append("text")
            if has_audio: inputs.append("audio")
            if has_images: inputs.append(f"{len(state.get('image_paths', []))} images")
            if has_docs: inputs.append(f"{len(state.get('document_paths', []))} documents")
            plan_msg += ", ".join(inputs) if inputs else "patient data"
            plan_msg += " → Ingest → Build Context → Check History → Find Similar Cases → Analyze → Review"
            
            await self._think(plan_msg)
            await self._log("[Orchestrator] Pipeline: Ingestion → Context → Memory → Retriever → Recommender → Reviewer")
            
            # ===================== INGESTION =====================
            await self._step("Ingestion", "active")
            await self._think("📥 Processing raw input data...")
            await self._log("[Ingestion Agent] Starting data ingestion...")
            
            input_data = FieldIngestionInput(
                patient_hash=state["patient_hash"],
                raw_text=state.get("raw_text"),
                audio_path=state.get("audio_path"),
                image_paths=state.get("image_paths"),
                document_paths=state.get("document_paths"),
                device_id=state.get("device_id", "device_001"),
                offline=state.get("offline", False)
            )
            
            output = self.field_ingestion.run(input_data)
            state["patient_data"] = output.patient_data.model_dump()
            
            for log in output.logs:
                await self._log(f"[Ingestion] {log}")
            
            await self._step("Ingestion", "complete")
            await self._log("[Ingestion Agent] Complete!", "success")
            
            # ===================== CONTEXT =====================
            await self._step("Context", "active")
            await self._log("[Context Agent] Building patient context...")
            
            patient_data = PatientData(**state["patient_data"])
            input_data = ContextBuilderInput(patient_data=patient_data)
            
            output = self.context_builder.run(input_data)
            state["context_payload"] = output.context_payload.model_dump()
            state["retrieval_plan"] = output.retrieval_plan.model_dump()
            
            for log in output.logs:
                await self._log(f"[Context] {log}")
            
            await self._step("Context", "complete")
            await self._log("[Context Agent] Complete!", "success")
            
            # ===================== MEMORY =====================
            await self._step("Memory", "active")
            await self._log("[Memory Agent] Processing patient history...")
            
            context_payload = ContextPayload(**state["context_payload"])
            input_data = PatientMemoryInput(
                patient_data=patient_data,
                context_payload=context_payload
            )
            
            output = self.patient_memory.run(input_data)
            state["patient_current_state"] = output.patient_current_state.model_dump()
            
            for log in output.logs:
                await self._log(f"[Memory] {log}")
            
            await self._step("Memory", "complete")
            await self._log("[Memory Agent] Complete!", "success")
            
            # ===================== RETRIEVER =====================
            await self._step("Retriever", "active")
            await self._log("[Retriever Agent] Searching similar cases...")
            
            retrieval_plan = RetrievalPlan(**state["retrieval_plan"])
            patient_current_state = PatientCurrentState(**state["patient_current_state"])
            
            input_data = RetrieverInput(
                retrieval_plan=retrieval_plan,
                patient_current_state=patient_current_state
            )
            
            output = self.retriever.run(input_data)
            state["retrieved_cases"] = [c.model_dump() for c in output.retrieved_cases]
            
            for log in output.logs:
                await self._log(f"[Retriever] {log}")
            
            await self._step("Retriever", "complete")
            await self._log(f"[Retriever Agent] Found {len(state['retrieved_cases'])} similar cases!", "success")
            
            # ===================== RECOMMENDER =====================
            await self._step("Recommender", "active")
            await self._log("[Recommender Agent] Generating recommendations...")
            
            retrieved_cases = [RetrievedCase(**c) for c in state.get("retrieved_cases", [])]
            input_data = RecommenderInput(
                patient_current_state=patient_current_state,
                retrieved_cases=retrieved_cases
            )
            
            output = self.recommender.run(input_data)
            state["case_analysis"] = output.analysis.model_dump()
            
            for log in output.logs:
                await self._log(f"[Recommender] {log}")
            
            await self._step("Recommender", "complete")
            await self._log("[Recommender Agent] Complete!", "success")
            
            # ===================== REVIEWER =====================
            await self._step("Reviewer", "active")
            await self._log("[Reviewer Agent] Reviewing analysis for safety...")
            
            case_analysis = CaseAnalysis(**state["case_analysis"])
            input_data = ReviewerInput(
                case_analysis=case_analysis,
                retrieved_cases=retrieved_cases
            )
            
            output = self.reviewer.run(input_data)
            state["review_verdict"] = output.verdict.model_dump()
            
            for log in output.logs:
                await self._log(f"[Reviewer] {log}")
            
            # Finalize
            state["final_output"] = {
                "patient_hash": state["patient_hash"],
                "analysis": state["case_analysis"],
                "verdict": state["review_verdict"],
                "approved": output.verdict.approved
            }
            
            verdict_emoji = "Approved" if output.verdict.approved else "Needs Review"
            await self._step("Reviewer", "complete")
            await self._log(f"[Reviewer Agent] Complete! Verdict: {verdict_emoji}", "success")
            
        except Exception as e:
            state["error"] = str(e)
            await self._log(f"[ERROR] Workflow failed: {e}", "error")
            logger.exception(f"Workflow error: {e}")
            
        return state


# For backward compatibility with existing code
def create_workflow():
    """Create a simple sync workflow (backward compatible)."""
    from langgraph.graph import StateGraph, END
    
    orchestrator_node = _SyncOrchestratorNode()
    
    workflow = StateGraph(WorkflowState)
    workflow.add_node("ingestion", orchestrator_node.ingestion_node)
    workflow.add_node("context", orchestrator_node.context_node)
    workflow.add_node("memory", orchestrator_node.memory_node)
    workflow.add_node("retriever", orchestrator_node.retriever_node)
    workflow.add_node("recommender", orchestrator_node.recommender_node)
    workflow.add_node("reviewer", orchestrator_node.reviewer_node)
    
    workflow.set_entry_point("ingestion")
    workflow.add_edge("ingestion", "context")
    workflow.add_edge("context", "memory")
    workflow.add_edge("memory", "retriever")
    workflow.add_edge("retriever", "recommender")
    workflow.add_edge("recommender", "reviewer")
    workflow.add_edge("reviewer", END)
    
    return workflow.compile()


class _SyncOrchestratorNode:
    """Sync orchestrator for backward compatibility."""
    
    def __init__(self):
        self.qdrant = QdrantManager()
        self.db = DatabaseManager()
        self.llm = GeminiClient()
        
        self.field_ingestion = FieldIngestionAgent(db_manager=self.db)
        self.context_builder = ContextBuilderAgent()
        self.patient_memory = PatientMemoryAgent(qdrant_manager=self.qdrant, db_manager=self.db)
        self.retriever = RetrieverAgent(qdrant_manager=self.qdrant)
        self.recommender = RecommenderAgent(llm_client=self.llm)
        self.reviewer = ReviewerAgent(llm_client=self.llm)
        
        logger.info("Orchestrator initialized successfully.")

    def ingestion_node(self, state: WorkflowState) -> WorkflowState:
        try:
            input_data = FieldIngestionInput(
                patient_hash=state["patient_hash"],
                raw_text=state.get("raw_text"),
                audio_path=state.get("audio_path"),
                image_paths=state.get("image_paths"),
                document_paths=state.get("document_paths"),
                device_id=state.get("device_id", "device_001"),
                offline=state.get("offline", False)
            )
            output = self.field_ingestion.run(input_data)
            state["patient_data"] = output.patient_data.model_dump()
            state["logs"].extend(output.logs)
        except Exception as e:
            state["error"] = f"Ingestion Failed: {e}"
        return state

    def context_node(self, state: WorkflowState) -> WorkflowState:
        try:
            patient_data = PatientData(**state["patient_data"])
            input_data = ContextBuilderInput(patient_data=patient_data)
            output = self.context_builder.run(input_data)
            state["context_payload"] = output.context_payload.model_dump()
            state["retrieval_plan"] = output.retrieval_plan.model_dump()
            state["logs"].extend(output.logs)
        except Exception as e:
            state["error"] = f"Context Build Failed: {e}"
        return state

    def memory_node(self, state: WorkflowState) -> WorkflowState:
        try:
            patient_data = PatientData(**state["patient_data"])
            context_payload = ContextPayload(**state["context_payload"])
            input_data = PatientMemoryInput(patient_data=patient_data, context_payload=context_payload)
            output = self.patient_memory.run(input_data)
            state["patient_current_state"] = output.patient_current_state.model_dump()
            state["logs"].extend(output.logs)
        except Exception as e:
            state["error"] = f"Patient Memory Failed: {e}"
        return state

    def retriever_node(self, state: WorkflowState) -> WorkflowState:
        try:
            retrieval_plan = RetrievalPlan(**state["retrieval_plan"])
            patient_current_state = PatientCurrentState(**state["patient_current_state"])
            input_data = RetrieverInput(retrieval_plan=retrieval_plan, patient_current_state=patient_current_state)
            output = self.retriever.run(input_data)
            state["retrieved_cases"] = [c.model_dump() for c in output.retrieved_cases]
            state["logs"].extend(output.logs)
        except Exception as e:
            state["error"] = f"Retriever Failed: {e}"
        return state

    def recommender_node(self, state: WorkflowState) -> WorkflowState:
        try:
            patient_current_state = PatientCurrentState(**state["patient_current_state"])
            retrieved_cases = [RetrievedCase(**c) for c in state.get("retrieved_cases", [])]
            input_data = RecommenderInput(patient_current_state=patient_current_state, retrieved_cases=retrieved_cases)
            output = self.recommender.run(input_data)
            state["case_analysis"] = output.analysis.model_dump()
            state["logs"].extend(output.logs)
        except Exception as e:
            state["error"] = f"Recommender Failed: {e}"
        return state

    def reviewer_node(self, state: WorkflowState) -> WorkflowState:
        try:
            case_analysis = CaseAnalysis(**state["case_analysis"])
            retrieved_cases = [RetrievedCase(**c) for c in state.get("retrieved_cases", [])]
            input_data = ReviewerInput(case_analysis=case_analysis, retrieved_cases=retrieved_cases)
            output = self.reviewer.run(input_data)
            state["review_verdict"] = output.verdict.model_dump()
            state["logs"].extend(output.logs)
            state["final_output"] = {
                "patient_hash": state["patient_hash"],
                "analysis": state["case_analysis"],
                "verdict": state["review_verdict"],
                "approved": output.verdict.approved
            }
        except Exception as e:
            state["error"] = f"Reviewer Failed: {e}"
        return state
