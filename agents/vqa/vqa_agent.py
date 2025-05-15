#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical VQA Agent
----------------
Agent chuyên biệt để trả lời câu hỏi về hình ảnh y tế sử dụng LLaVA-Med.
"""

import os
import sys
import json
import logging
import time
import torch
from PIL import Image
import requests
from io import BytesIO
from dataclasses import dataclass, field
from typing import Dict, List, Union, Any, Optional, Tuple

from agents.base_agent import BaseAgent, BaseAgentConfig

@dataclass
class VQAAgentConfig(BaseAgentConfig):
    """Cấu hình cho VQA Agent."""
    # Tham số model
    model_path: str
    name: str = "MedicalVQAAgent"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.7
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Prompt engineering
    prompt_template: str = "I am a helpful medical AI assistant. I will answer your question about this medical image based on the findings I can observe.\n\nQuestion: {question}\n\nAnswer:"
    system_prompt: str = "You are a medical imaging expert with years of experience. Analyze the image carefully and provide accurate, detailed answers based solely on what you can observe."
    
    # Các prompt template chuyên biệt cho các loại câu hỏi khác nhau
    prompt_templates: Dict[str, str] = field(default_factory=lambda: {
        "general": "I am a helpful medical AI assistant. I will answer your question about this medical image based on the findings I can observe.\n\nQuestion: {question}\n\nAnswer:",
        "diagnosis": "I am a medical imaging specialist with expertise in diagnosis. I will analyze this medical image carefully and provide my diagnostic assessment based on visible findings.\n\nClinical question: {question}\n\nDiagnostic interpretation:",
        "description": "I am a medical imaging expert. I will describe this medical image in detail, noting all relevant anatomical structures and abnormalities visible.\n\nRequest: {question}\n\nImage description:",
        "comparison": "I am a medical imaging specialist. I will compare and contrast the findings in this medical image based on the question.\n\nComparison request: {question}\n\nComparative analysis:"
    })
    
    # Từ ngữ chỉ độ tin cậy thấp
    low_confidence_phrases: List[str] = field(default_factory=lambda: [
        "i'm not sure", "i am not sure", "unclear", "cannot determine",
        "difficult to say", "hard to tell", "cannot see", "not visible",
        "may be", "might be", "possibly", "probably", "uncertain", 
        "i cannot", "i can't", "limited visibility", "poor quality",
        "without additional", "would need more", "based solely on"
    ])
    
    # Xử lý ảnh
    image_size: Tuple[int, int] = (336, 336)  # Kích thước ảnh input cho LLaVA-Med
    normalize_image: bool = True              # Chuẩn hóa ảnh
    
    # Caching
    enable_cache: bool = True                # Bật cache cho câu trả lời
    cache_expiry: int = 3600                 # Thời gian hết hạn cache (giây)


class MedicalVQAAgent(BaseAgent):
    """
    Agent trả lời câu hỏi dựa trên hình ảnh y tế sử dụng LLaVA-Med.
    
    Đặc điểm:
    - Xử lý hình ảnh y tế và trả lời câu hỏi về nội dung
    - Tối ưu hóa prompt cho từng loại câu hỏi
    - Đánh giá độ tin cậy của câu trả lời
    - Sử dụng memory để cải thiện trải nghiệm
    """
    
    def __init__(self, config: VQAAgentConfig):
        """
        Khởi tạo Medical VQA Agent.
        
        Args:
            config: Cấu hình cho agent
        """
        super().__init__(config)
        self.vqa_config = config
        self.device = torch.device(config.device)
        
        # Các thành phần LLaVA-Med
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.conv = None
        
        # Cache kết quả
        self.result_cache = {}
        
        # Metrics bổ sung
        self.metrics.update({
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_confidence": 0,
            "total_confidence": 0,
            "prompt_types_used": {
                "general": 0,
                "diagnosis": 0,
                "description": 0,
                "comparison": 0
            }
        })
        
        self.logger.info(f"VQA Agent đã được khởi tạo với cấu hình: {self.vqa_config}")
    
    def initialize(self) -> bool:
        """
        Khởi tạo VQA Agent, tải model LLaVA-Med.
        """
        try:
            self.logger.info(f"Đang tải model LLaVA-Med từ thư mục {self.vqa_config.model_path}")
            
            # Kiểm tra thư mục model
            if not os.path.exists(self.vqa_config.model_path):
                self.logger.warning(f"Không tìm thấy thư mục model tại {self.vqa_config.model_path}")
            
            # Import các thư viện cần thiết
            try:
                from llava.model.builder import load_pretrained_model
                from llava.mm_utils import get_model_name_from_path
                from llava.conversation import conv_templates
                
                # Giả định model_name từ tên thư mục
                model_name = os.path.basename(self.vqa_config.model_path.rstrip('/'))
                
                # Tải model LLaVA-Med từ thư mục
                self.tokenizer, self.model, self.image_processor, self.context_len = \
                    load_pretrained_model(self.vqa_config.model_path, model_name, self.device)
                
                # Cài đặt template hội thoại
                self.conv = conv_templates["llava_v1"].copy()
                
                self.logger.info(f"Model LLaVA-Med đã được tải thành công trên thiết bị {self.device}")
                self.initialized = True
                return True
                    
            except ImportError as e:
                self.logger.error(f"Không thể import các module cần thiết: {str(e)}")
                self.logger.error("Vui lòng cài đặt llava-med và các dependency: pip install llava")
                return False
                    
        except Exception as e:
            self.logger.error(f"Lỗi khi khởi tạo VQA Agent: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý yêu cầu VQA.
        
        Args:
            input_data: Dictionary chứa:
                - image_path: Đường dẫn đến hình ảnh hoặc URL
                - query: Câu hỏi về hình ảnh
                - context (tùy chọn): Ngữ cảnh bổ sung
                
        Returns:
            Dictionary chứa kết quả và metadata
        """
        if not self.initialized:
            return {"error": "VQA Agent chưa được khởi tạo", "success": False}
        
        # Trích xuất dữ liệu đầu vào
        image_path = input_data.get("image_path")
        query = input_data.get("query")
        context = input_data.get("context", {})
        
        # Kiểm tra đầu vào
        if not image_path:
            return {"error": "Thiếu đường dẫn hình ảnh (image_path)", "success": False}
        if not query:
            return {"error": "Thiếu câu hỏi (query)", "success": False}
        
        # Tạo cache key nếu caching được bật
        cache_key = None
        if self.vqa_config.enable_cache:
            cache_key = f"{image_path}:{query}"
            cached_result = self._check_cache(cache_key)
            if cached_result:
                self.metrics["cache_hits"] += 1
                return cached_result
            self.metrics["cache_misses"] += 1
        
        # Thực hiện quá trình xử lý với monitoring
        result = self.run_with_monitoring(self._process_vqa_request, image_path, query, context)
        
        # Lưu vào cache nếu cần
        if self.vqa_config.enable_cache and cache_key and "error" not in result:
            self._add_to_cache(cache_key, result)
        
        # Lưu vào bộ nhớ ngắn hạn
        self.remember("last_vqa_result", result)
        self.remember(f"vqa_result_{image_path}", result)
        
        return result
    
    def _process_vqa_request(self, image_path: Union[str, Image.Image], query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý nội bộ cho yêu cầu VQA.
        
        Args:
            image_path: Đường dẫn đến hình ảnh hoặc PIL Image
            query: Câu hỏi về hình ảnh
            context: Ngữ cảnh bổ sung
            
        Returns:
            Dictionary chứa câu trả lời và metadata
        """
        try:
            # Tiền xử lý hình ảnh
            self.logger.info(f"Đang tiền xử lý hình ảnh: {image_path}")
            image_tensor = self.preprocess_image(image_path)
            
            # Tối ưu hóa prompt
            prompt_type, enhanced_query = self._optimize_prompt(query, context)
            
            # Cập nhật metrics
            self.metrics["prompt_types_used"][prompt_type] += 1
            
            # Ghi log prompt đã tối ưu
            self.logger.debug(f"Prompt đã tối ưu ({prompt_type}): {enhanced_query}")
            
            # Sinh câu trả lời
            result = self._generate_answer(image_tensor, enhanced_query)
            
            # Đánh giá độ tin cậy
            confidence = self._estimate_confidence(result)
            
            # Cập nhật metrics confidence
            self.metrics["total_confidence"] += confidence
            self.metrics["avg_confidence"] = self.metrics["total_confidence"] / self.metrics["total_requests"]
            
            # Đóng gói kết quả
            return {
                "answer": result,
                "confidence": confidence,
                "high_confidence": confidence >= self.vqa_config.confidence_threshold,
                "prompt_type": prompt_type,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý VQA: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "error": f"Không thể xử lý yêu cầu: {str(e)}",
                "success": False
            }
    
    def preprocess_image(self, image_path: Union[str, Image.Image]) -> torch.Tensor:
        """
        Tiền xử lý hình ảnh cho model.
        
        Args:
            image_path: Đường dẫn đến hình ảnh hoặc PIL Image
            
        Returns:
            torch.Tensor: Tensor hình ảnh đã xử lý
        """
        try:
            # Xử lý các loại đầu vào khác nhau
            if isinstance(image_path, str):
                if image_path.startswith(('http://', 'https://')):
                    # Tải hình ảnh từ URL
                    response = requests.get(image_path)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    # Tải hình ảnh từ đường dẫn cục bộ
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Không tìm thấy file hình ảnh: {image_path}")
                    image = Image.open(image_path).convert('RGB')
            else:
                # Đã là PIL Image
                image = image_path.convert('RGB')
            
            # Resize nếu cấu hình yêu cầu
            if self.vqa_config.image_size:
                image = image.resize(self.vqa_config.image_size, Image.LANCZOS)
            
            # Xử lý với image processor
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tiền xử lý hình ảnh: {str(e)}")
            raise ValueError(f"Không thể xử lý hình ảnh: {str(e)}")
    
    def _optimize_prompt(self, query: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """
        Tối ưu hóa prompt dựa trên câu hỏi và ngữ cảnh.
        
        Args:
            query: Câu hỏi ban đầu
            context: Ngữ cảnh bổ sung
            
        Returns:
            Tuple[str, str]: (loại prompt, prompt đã tối ưu)
        """
        # Chọn loại prompt phù hợp dựa trên từ khóa
        prompt_type = "general"
        
        # Phân tích câu hỏi để xác định loại prompt phù hợp
        lower_query = query.lower()
        if any(kw in lower_query for kw in ["diagnos", "condition", "disease", "abnormal", "pathology"]):
            prompt_type = "diagnosis"
        elif any(kw in lower_query for kw in ["describe", "what do you see", "show", "identify", "what is in"]):
            prompt_type = "description"
        elif any(kw in lower_query for kw in ["compare", "difference", "versus", "vs", "contrast"]):
            prompt_type = "comparison"
        
        # Lấy template tương ứng
        template = self.vqa_config.prompt_templates.get(prompt_type, self.vqa_config.prompt_template)
        
        # Chuẩn bị các tham số để render template
        template_params = {"question": query}
        
        # Thêm ngữ cảnh vào prompt nếu có
        context_info = []
        
        # Thêm thông tin từ detection nếu có
        detections = context.get("detections", [])
        if detections:
            detection_texts = []
            for i, det in enumerate(detections[:5]):  # Chỉ lấy 5 detection đầu tiên
                if isinstance(det, dict):
                    cls = det.get("class", "unknown")
                    conf = det.get("confidence", 0)
                    detection_texts.append(f"{cls} ({conf:.2f})")
                else:
                    detection_texts.append(f"object {i+1}")
            
            if detection_texts:
                context_info.append(f"Objects detected in image: {', '.join(detection_texts)}")
        
        # Thêm thông tin phân loại nếu có
        classifications = context.get("classifications", [])
        if classifications:
            class_texts = []
            for i, cls in enumerate(classifications[:3]):  # Chỉ lấy 3 phân loại đầu tiên
                if isinstance(cls, dict):
                    name = cls.get("class", "unknown")
                    conf = cls.get("confidence", 0)
                    class_texts.append(f"{name} ({conf:.2f})")
                else:
                    class_texts.append(f"class {i+1}")
            
            if class_texts:
                context_info.append(f"Image classified as: {', '.join(class_texts)}")
        
        # Thêm thông tin y tế bổ sung nếu có
        medical_context = context.get("medical_context")
        if medical_context:
            context_info.append(f"Medical context: {medical_context}")
        
        # Thêm patient_info nếu có
        patient_info = context.get("patient_info")
        if patient_info:
            context_info.append(f"Patient information: {patient_info}")
        
        # Kết hợp tất cả ngữ cảnh
        if context_info:
            template_params["context"] = "\n".join(context_info)
            # Thêm ngữ cảnh vào template
            enhanced_template = template.replace("Question:", "Context:\n{context}\n\nQuestion:")
        else:
            enhanced_template = template
        
        # Render template với tham số
        try:
            enhanced_query = enhanced_template.format(**template_params)
        except KeyError as e:
            self.logger.warning(f"Thiếu tham số khi render template: {e}")
            # Fallback to simple template
            enhanced_query = f"{self.vqa_config.prompt_template.format(question=query)}"
        
        return prompt_type, enhanced_query
    
    def _generate_answer(self, image_tensor: torch.Tensor, prompt: str) -> str:
        """
        Sinh câu trả lời cho hình ảnh và prompt.
        
        Args:
            image_tensor: Tensor hình ảnh đã xử lý
            prompt: Prompt đã tối ưu
            
        Returns:
            str: Câu trả lời
        """
        self.logger.info("Đang sinh câu trả lời")
        
        # Clear conversation history
        self.conv.clear()
        
        # Thêm prompt vào hội thoại
        self.conv.append_message(self.conv.roles[0], prompt)
        self.conv.append_message(self.conv.roles[1], None)
        
        # Get prompt
        prompt = self.conv.get_prompt()
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Generate with model
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.vqa_config.temperature,
                max_new_tokens=self.vqa_config.max_new_tokens,
                top_p=self.vqa_config.top_p,
                top_k=self.vqa_config.top_k,
                repetition_penalty=self.vqa_config.repetition_penalty,
            )
        
        # Decode output
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        self.conv.messages[-1][-1] = outputs
        
        return outputs.strip()
    
    def _estimate_confidence(self, answer: str) -> float:
        """
        Đánh giá độ tin cậy của câu trả lời.
        
        Args:
            answer: Câu trả lời từ model
            
        Returns:
            float: Điểm độ tin cậy từ 0.0 đến 1.0
        """
        # Chuyển sang chữ thường để dễ so sánh
        lower_answer = answer.lower()
        
        # Khởi tạo độ tin cậy cao
        confidence = 1.0
        
        # Kiểm tra các cụm từ biểu thị độ tin cậy thấp
        for phrase in self.vqa_config.low_confidence_phrases:
            if phrase in lower_answer:
                # Giảm độ tin cậy cho mỗi cụm từ tìm thấy
                confidence -= 0.1
                
                # Log để debug
                self.logger.debug(f"Tìm thấy cụm từ độ tin cậy thấp: '{phrase}'")
                
                # Giới hạn dưới cho độ tin cậy
                if confidence < 0.3:
                    confidence = 0.3
                    break
        
        # Kiểm tra chiều dài câu trả lời - câu trả lời quá ngắn có thể kém tin cậy
        if len(answer.split()) < 10:
            confidence -= 0.1
            self.logger.debug("Câu trả lời quá ngắn, giảm độ tin cậy")
        
        # Kiểm tra tính không chắc chắn qua cách dùng từ
        uncertainty_indicators = [
            "cannot rule out", "differential diagnosis", "possibilities include",
            "could be", "several possibilities", "not conclusive", "not definitive"
        ]
        for indicator in uncertainty_indicators:
            if indicator in lower_answer:
                confidence -= 0.05
                self.logger.debug(f"Tìm thấy dấu hiệu không chắc chắn: '{indicator}'")
        
        return max(0.0, min(1.0, confidence))  # Đảm bảo confidence nằm trong khoảng [0.0, 1.0]
    
    def _check_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Kiểm tra cache cho kết quả đã tính toán trước đó.
        
        Args:
            key: Khóa cache
            
        Returns:
            Optional[Dict]: Kết quả cache hoặc None nếu không tìm thấy
        """
        if not self.vqa_config.enable_cache:
            return None
            
        cached_item = self.result_cache.get(key)
        if not cached_item:
            return None
            
        # Kiểm tra thời gian hết hạn
        if time.time() - cached_item["timestamp"] > self.vqa_config.cache_expiry:
            # Cache đã hết hạn
            del self.result_cache[key]
            return None
            
        self.logger.info(f"Đã tìm thấy kết quả trong cache cho key: {key}")
        return cached_item["result"]
    
    def _add_to_cache(self, key: str, result: Dict[str, Any]):
        """
        Thêm kết quả vào cache.
        
        Args:
            key: Khóa cache
            result: Kết quả cần cache
        """
        if not self.vqa_config.enable_cache:
            return
            
        self.result_cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        # Log cache stats
        self.logger.debug(f"Cache hiện tại có {len(self.result_cache)} mục")
    
    def clear_cache(self):
        """Xóa tất cả các mục trong cache."""
        cache_size = len(self.result_cache)
        self.result_cache.clear()
        self.logger.info(f"Đã xóa {cache_size} mục trong cache")
    
    def batch_process(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Xử lý hàng loạt các yêu cầu VQA.
        
        Args:
            batch_data: Danh sách các yêu cầu VQA
            
        Returns:
            List[Dict]: Danh sách kết quả
        """
        results = []
        
        self.logger.info(f"Đang xử lý hàng loạt {len(batch_data)} yêu cầu")
        
        for request in batch_data:
            # Xử lý từng yêu cầu
            result = self.process(request)
            results.append(result)
        
        return results