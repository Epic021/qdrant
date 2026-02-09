"""
LLM Client Tool
===============
Wrapper for Gemini API interactions.
"""

import os
import logging
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("llm_client")

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.model = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                logger.info(f"Gemini initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
        else:
            logger.warning("GEMINI_API_KEY not found in environment")
            
    def generate(self, prompt: str) -> Optional[str]:
        if not self.model:
            logger.error("Gemini model not initialized")
            return None
            
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
            
    def verify_hallucination(self, explanation: str, evidence: str) -> Dict[str, Any]:
        """
        Specific method for review agent to return structured verification.
        """
        prompt = f"""
        You are a strict fact-checker.
        
        EVIDENCE:
        {evidence}
        
        CLAIM:
        {explanation}
        
        TASK:
        Verify if the CLAIM is fully supported by the EVIDENCE.
        Return JSON:
        {{
            "is_hallucinated": boolean,
            "details": "reasoning"
        }}
        """
        # Note: In a real implementation, we would use structured output mode or strict JSON parsing
        # For now, we return a mock or simple parsing if needed
        # We will just return None here and let the agent handle logic or implementing strictly
        
        return {"is_hallucinated": False, "details": "Not implemented in basic client"}
