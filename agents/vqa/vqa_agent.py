from agents.base_agent import BaseAgent
from tools.util_tools import VisualizationTool
from tools.vqa_tools import LLaVATool

class VQAAgent(BaseAgent):
    """
    Agent for answering questions about medical endoscopy images.
    Uses LLaVA model for visual question answering with reasoning.
    """
    
    def __init__(self, llm_client, model_path: str, device: str = "cuda"):
        """Initialize the VQA agent."""
        
        # System prompt
        system_prompt = """
        You are a Medical Visual Question Answering Agent specialized in analyzing and explaining endoscopy images.
        
        Your capabilities:
        1. You can answer questions about what is visible in endoscopy images
        2. You can identify and describe polyps, lesions, and anatomical structures
        3. You can interpret the clinical significance of visual findings
        4. You can use detection results and classifications to enhance your answers
        
        Important medical context:
        - Endoscopy images require careful interpretation by trained professionals
        - Your analysis is meant to assist, not replace, a clinician's judgment
        - Polyp characteristics like size, shape, and surface pattern are clinically significant
        - Different GI tract regions have distinct appearances and common pathologies
        
        Task approach:
        1. Use the LLaVA tool to analyze the image and answer the question
        2. If available, consider detection and classification results for context
        3. Provide a clear, medically accurate answer
        4. Acknowledge uncertainty when appropriate
        
        Be concise but precise in your answers. Focus on clinically relevant information and avoid speculation beyond what is visible in the image.
        """
        
        # Initialize base agent
        super().__init__(
            name="Medical VQA Agent",
            system_prompt=system_prompt,
            llm_client=llm_client,
            tools=[],  # Will add tools after initialization
            memory_enabled=True,
            device=device
        )
        
        # Add LLaVA tool
        llava_tool = LLaVATool(model_path=model_path, device=device)
        self.register_tool(llava_tool)
        
        # Add visualization tool
        vis_tool = VisualizationTool()
        self.register_tool(vis_tool)