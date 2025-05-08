import logging
from typing import Dict, List, Union, Any
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MedRAGAgent")

@dataclass
class RAGConfig:
    """Configuration for the Medical RAG Agent."""
    llm_model_path: str
    embedding_model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    num_retrieved_docs: int = 3

class MedicalRAGAgent:
    """
    A Retrieval-Augmented Generation (RAG) agent specialized for medical knowledge
    retrieval and question answering.
    """
    
    def __init__(self, config: RAGConfig):
        """Initialize the Medical RAG Agent."""
        self.config = config
        self.device = torch.device(config.device)
        logger.info(f"Initializing Medical RAG Agent on device: {self.device}")
        
        # Load models
        self._load_models()
        
        # Initialize metrics
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0
        }

    def _load_models(self):
        """Load the language model and embedding model."""
        try:
            logger.info("Loading language model and embedding model")
            
            # Load language model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.config.embedding_model_path)
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise RuntimeError(f"Failed to load models: {str(e)}")

    def answer_question(self, question: str, context: List[str] = None) -> Dict[str, Any]:
        """
        Answer a medical question using RAG.
        
        Args:
            question: The question to answer
            context: Optional list of context documents
            
        Returns:
            Dict containing answer and metadata
        """
        logger.info(f"Processing question: {question}")
        
        try:
            # Retrieve relevant documents if context not provided
            if context is None:
                context = self._retrieve_documents(question)
            
            # Generate answer
            answer = self._generate_answer(question, context)
            
            # Update metrics
            self._update_metrics(True)
            
            return {
                "answer": answer,
                "context": context,
                "metrics": self.metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            self._update_metrics(False)
            
            return {
                "error": str(e),
                "answer": None,
                "context": None
            }

    def _retrieve_documents(self, question: str) -> List[str]:
        """Retrieve relevant documents for the question."""
        # Implement your document retrieval logic here
        # This is a placeholder implementation
        return []

    def _generate_answer(self, question: str, context: List[str]) -> str:
        """Generate an answer using the language model."""
        # Implement your answer generation logic here
        # This is a placeholder implementation
        return ""

    def _update_metrics(self, success: bool):
        """Update the agent's performance metrics."""
        self.metrics["total_queries"] += 1
        if success:
            self.metrics["successful_queries"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get the agent's performance metrics."""
        return self.metrics

    def export_metrics(self, output_path: str):
        """Export the agent's metrics to a file."""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics exported to {output_path}") 