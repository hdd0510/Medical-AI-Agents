import torch
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Union
import os
from tools.base_tools import Tool
class LLaVATool(Tool):
    """Tool for visual question answering using LLaVA models."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__(
            name="llava_vqa",
            description="Answer questions about medical images using LLaVA (Large Language and Vision Assistant)."
        )
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.conv = None
        self._load_model()
        
    def _load_model(self):
        """Load the LLaVA model."""
        try:
            # Import LLaVA components
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.conversation import conv_templates
            
            # Get model name
            model_name = os.path.basename(self.model_path.rstrip('/'))
            
            # Load model
            self.tokenizer, self.model, self.image_processor, self.context_len = \
                load_pretrained_model(self.model_path, model_name, self.device)
            
            # Set conversation template
            self.conv = conv_templates["llava_v1"].copy()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LLaVA model: {str(e)}")
    
    def run(self, 
           image_path: str, 
           query: str,
           temperature: float = 0.2,
           max_new_tokens: int = 512,
           prompt_template: str = None) -> Dict[str, Any]:
        """
        Run VQA on an image.
        
        Args:
            image_path: Path to the image
            query: Question about the image
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate
            prompt_template: Optional custom prompt template
            
        Returns:
            Dict with VQA results
        """
        if not self.model:
            return {"error": "Model not loaded", "success": False}
            
        try:
            # Load and preprocess image
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    return {"error": f"Image path not found: {image_path}", "success": False}
                image = Image.open(image_path).convert("RGB")
            else:
                image = image_path
            
            # Preprocess image
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            image_tensor = image_tensor.to(self.device)
            
            # Prepare prompt
            if prompt_template is None:
                prompt_template = "I am a medical AI assistant specialized in analyzing medical images. I'll answer your question about this medical image based on what I can observe.\n\nQuestion: {question}\n\nAnswer:"
            
            prompt = prompt_template.format(question=query)
            
            # Clear conversation history
            self.conv.clear()
            
            # Add prompt to conversation
            self.conv.append_message(self.conv.roles[0], prompt)
            self.conv.append_message(self.conv.roles[1], None)
            
            # Get prompt
            prompt = self.conv.get_prompt()
            
            # Tokenize
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens
                )
            
            # Decode output
            outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            answer = outputs.strip()
            
            # Estimate confidence based on answer patterns
            confidence = self._estimate_confidence(answer)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "prompt": prompt,
                "image_path": image_path if isinstance(image_path, str) else "in-memory-image",
                "success": True
            }
                
        except Exception as e:
            import traceback
            return {
                "error": f"VQA failed: {str(e)}",
                "traceback": traceback.format_exc(),
                "success": False
            }
    
    def _estimate_confidence(self, answer: str) -> float:
        """Estimate confidence based on answer patterns."""
        # Simple heuristic - can be improved
        low_confidence_phrases = [
            "i'm not sure", "i am not sure", "unclear", "cannot determine",
            "difficult to say", "hard to tell", "cannot see", "not visible",
            "may be", "might be", "possibly", "probably", "uncertain"
        ]
        
        answer_lower = answer.lower()
        confidence = 1.0
        
        # Reduce confidence for uncertainty phrases
        for phrase in low_confidence_phrases:
            if phrase in answer_lower:
                confidence -= 0.1
                if confidence < 0.3:
                    confidence = 0.3
                    break
        
        # Reduce confidence for very short answers
        if len(answer.split()) < 10:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get the parameters schema for this tool."""
        return {
            "image_path": {
                "type": "string",
                "description": "Path to the medical image"
            },
            "query": {
                "type": "string",
                "description": "Question about the medical image"
            },
            "temperature": {
                "type": "number",
                "description": "Sampling temperature (0.0 for deterministic)",
                "default": 0.2
            },
            "max_new_tokens": {
                "type": "integer",
                "description": "Maximum new tokens to generate",
                "default": 512
            },
            "prompt_template": {
                "type": "string",
                "description": "Optional custom prompt template with {question} placeholder",
                "default": None
            }
        }
    
    def _get_returns(self) -> Dict[str, Any]:
        """Get the return schema for this tool."""
        return {
            "answer": {
                "type": "string",
                "description": "Answer to the question about the image"
            },
            "confidence": {
                "type": "number",
                "description": "Estimated confidence score (0.0-1.0)"
            },
            "success": {
                "type": "boolean",
                "description": "Whether the VQA was successful"
            }
        }