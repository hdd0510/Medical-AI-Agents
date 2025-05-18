from agents.base_agent import BaseAgent
from tools.util_tools import VisualizationTool
from tools.det_tools import YOLOTool

class DetectorAgent(BaseAgent):
    """
    Agent for detecting polyps and other objects in medical endoscopy images.
    Combines YOLO-based detection with reasoning capabilities.
    """
    
    def __init__(self, llm_client, model_path: str, device: str = "cuda"):
        """Initialize the detector agent."""
        
        # System prompt
        system_prompt = """
        You are a Medical Detector Agent specialized in detecting polyps and abnormalities in endoscopy images.
        
        Your capabilities:
        1. You can detect polyps, lesions, and abnormalities in gastrointestinal endoscopy images
        2. You can analyze detection results and provide clinical interpretation
        3. You can visualize detections with bounding boxes and annotations
        
        Important medical context:
        - Polyps are abnormal tissue growths that may be precancerous
        - Early detection is critical for preventing colorectal cancer
        - Different types of polyps include: adenomatous, hyperplastic, and serrated
        - Size and appearance are important factors in risk assessment
        
        Task approach:
        1. Always use the detection tool first to identify objects in the image
        2. Analyze the detection results (count, confidence, size, location)
        3. If needed, visualize the detections for better interpretation
        4. Provide clinically relevant insights based on detection results
        
        Be concise but precise in your answers. Focus on clear communication of detection results and their clinical significance.
        """
        
        # Initialize base agent
        super().__init__(
            name="Medical Detector Agent",
            system_prompt=system_prompt,
            llm_client=llm_client,
            tools=[],  # Will add tools after initialization
            memory_enabled=True,
            device=device
        )
        
        # Add YOLO detection tool
        yolo_tool = YOLOTool(model_path=model_path, device=device)
        self.register_tool(yolo_tool)
        
        # Add visualization tool
        vis_tool = VisualizationTool()
        self.register_tool(vis_tool)