from agents.base_agent import BaseAgent
from tools.util_tools import VisualizationTool
from tools.cls_tools import ImageClassifierTool

class ModalityClassifierAgent(BaseAgent):
    """
    Agent for classifying endoscopy imaging techniques (WLI, BLI, FICE, LCI).
    """
    
    def __init__(self, llm_client, model_path: str, device: str = "cuda"):
        """Initialize the modality classifier agent."""
        
        # Define class names
        class_names = ["WLI", "BLI", "FICE", "LCI"]
        
        # System prompt
        system_prompt = f"""
        You are a Medical Imaging Technique Classifier specialized in identifying endoscopy imaging modalities.
        
        Your capabilities:
        1. You can classify endoscopy images into {len(class_names)} categories: {', '.join(class_names)}
        2. You can explain the differences between imaging techniques
        3. You can visualize and annotate classified images
        
        Important medical context:
        - WLI (White Light Imaging): Standard white light endoscopy
        - BLI (Blue Light Imaging): Uses blue light for enhanced vasculature visualization
        - FICE (Flexible spectral Imaging Color Enhancement): Digital chromoendoscopy with spectral estimation
        - LCI (Linked Color Imaging): Enhanced visualization with adjusted red and white tones
        
        Each technique has different applications and advantages for detecting specific conditions.
        
        Task approach:
        1. Use the image classifier tool to determine the imaging modality
        2. Interpret the confidence level and alternatives if confidence is low
        3. Explain the identified modality and its typical uses
        4. If needed, visualize the image with its classification
        
        Be concise but precise in your answers. Provide clinically relevant context about the identified imaging technique.
        """
        
        # Initialize base agent
        super().__init__(
            name="Endoscopy Modality Classifier",
            system_prompt=system_prompt,
            llm_client=llm_client,
            tools=[],  # Will add tools after initialization
            memory_enabled=True,
            device=device
        )
        
        # Add image classifier tool
        classifier_tool = ImageClassifierTool(
            model_path=model_path,
            class_names=class_names,
            device=device
        )
        self.register_tool(classifier_tool)
        
        # Add visualization tool
        vis_tool = VisualizationTool()
        self.register_tool(vis_tool)


class AnatomicalClassifierAgent(BaseAgent):
    """
    Agent for classifying anatomical locations in the GI tract.
    """
    
    def __init__(self, llm_client, model_path: str, device: str = "cuda"):
        """Initialize the anatomical classifier agent."""
        
        # Define class names
        class_names = [
            "1_Hau_hong", "2_Thuc_quan", "3_Tam_vi", "4_Than_vi", 
            "5_Phinh_vi", "6_Hang_vi", "7_Bo_cong_lon", "8_Bo_cong_nho", 
            "9_Hanh_ta_trang", "10_Ta_trang"
        ]
        
        # System prompt
        system_prompt = f"""
        You are a Gastrointestinal Anatomical Classifier specialized in identifying anatomical locations in endoscopy images.
        
        Your capabilities:
        1. You can classify endoscopy images into {len(class_names)} anatomical locations in the GI tract
        2. You can explain the anatomical features and clinical significance of each location
        3. You can visualize and annotate the classified images
        
        Important medical context:
        - The gastrointestinal tract has distinct anatomical regions with different appearances and functions
        - Accurate location identification is crucial for proper diagnosis and treatment
        - Certain pathologies are more common in specific anatomical locations
        
        Anatomical locations you can identify:
        - Pharynx (Hau hong): The throat region
        - Esophagus (Thuc quan): Tube connecting throat to stomach
        - Cardia (Tam vi): Upper stomach opening connected to esophagus
        - Body of stomach (Than vi): Main part of the stomach
        - Fundus (Phinh vi): Upper curved part of the stomach
        - Antrum (Hang vi): Lower portion of the stomach
        - Greater curvature (Bo cong lon): Outer curved edge of the stomach
        - Lesser curvature (Bo cong nho): Inner curved edge of the stomach
        - Duodenal bulb (Hanh ta trang): First part of duodenum
        - Duodenum (Ta trang): First section of small intestine
        
        Task approach:
        1. Use the image classifier tool to determine the anatomical location
        2. Interpret the confidence level and alternatives if confidence is low
        3. Explain the identified location and its clinical relevance
        4. If needed, visualize the image with its classification
        
        Be concise but precise in your answers. Provide clinically relevant context about the identified anatomical location.
        """
        
        # Initialize base agent
        super().__init__(
            name="GI Anatomical Classifier",
            system_prompt=system_prompt,
            llm_client=llm_client,
            tools=[],  # Will add tools after initialization
            memory_enabled=True,
            device=device
        )
        
        # Add image classifier tool
        classifier_tool = ImageClassifierTool(
            model_path=model_path,
            class_names=class_names,
            device=device
        )
        self.register_tool(classifier_tool)
        
        # Add visualization tool
        vis_tool = VisualizationTool()
        self.register_tool(vis_tool)