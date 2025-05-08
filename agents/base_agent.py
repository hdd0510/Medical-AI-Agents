#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Agent
----------
Lớp trừu tượng cơ sở cho tất cả các agent trong hệ thống y tế AI.
"""

import os
import sys
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

@dataclass
class BaseAgentConfig:
    """Cấu hình cơ sở cho tất cả các agent."""
    name: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logging_level: str = "INFO"
    metrics_enabled: bool = True
    memory_enabled: bool = True
    inference_monitoring: bool = True
    confidence_threshold: float = 0.7
    
    # Đường dẫn cho các model và tài nguyên
    model_path: str = None
    resources_path: str = None
    
    # Các tham số khác
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Kiểm tra và thiết lập sau khi khởi tạo."""
        # Thiết lập logging level
        numeric_level = getattr(logging, self.logging_level, None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid logging level: {self.logging_level}")


class BaseAgent(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các agent trong hệ thống.
    
    Định nghĩa interface chung và chức năng mà tất cả các agent phải triển khai.
    """
    
    def __init__(self, config: BaseAgentConfig):
        """
        Khởi tạo agent cơ sở.
        
        Args:
            config: Cấu hình cho agent
        """
        self.config = config
        self.name = config.name
        self.device = config.device
        self.logger = logging.getLogger(f"agent.{self.name.lower().replace(' ', '_')}")
        self.logger.setLevel(getattr(logging, config.logging_level))
        
        # Trạng thái khởi tạo
        self.initialized = False
        
        # Bộ nhớ ngắn hạn cho agent
        self.short_term_memory = {}
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0,
            "avg_processing_time": 0,
            "high_confidence_count": 0,
            "low_confidence_count": 0
        }
        
        self.logger.info(f"Khởi tạo {self.name} với cấu hình: {self.config}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Khởi tạo agent, tải model và các tài nguyên cần thiết.
        
        Returns:
            bool: True nếu khởi tạo thành công, False nếu thất bại
        """
        pass
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xử lý dữ liệu đầu vào và trả về kết quả.
        
        Args:
            input_data: Dict chứa dữ liệu đầu vào cụ thể cho agent
            
        Returns:
            Dict chứa kết quả xử lý
        """
        pass
    
    def run_with_monitoring(self, func, *args, **kwargs) -> Dict[str, Any]:
        """
        Chạy một hàm với giám sát thời gian và thành công.
        
        Args:
            func: Hàm cần chạy
            *args: Tham số vị trí cho hàm
            **kwargs: Tham số từ khóa cho hàm
            
        Returns:
            Kết quả của hàm
        """
        start_time = time.time()
        success = True
        result = None
        
        try:
            # Chạy hàm
            result = func(*args, **kwargs)
            
            # Đánh giá độ tin cậy nếu có
            if isinstance(result, dict) and "confidence" in result:
                confidence = result["confidence"]
                if confidence >= self.config.confidence_threshold:
                    self.metrics["high_confidence_count"] += 1
                else:
                    self.metrics["low_confidence_count"] += 1
                    
                    # Log cảnh báo nếu độ tin cậy thấp
                    self.logger.warning(f"Kết quả có độ tin cậy thấp: {confidence}")
                    
                    # Đánh dấu trong kết quả
                    result["low_confidence_warning"] = True
            
        except Exception as e:
            success = False
            self.logger.error(f"Lỗi trong {func.__name__}: {str(e)}")
            result = {"error": str(e), "success": False}
            
        finally:
            # Cập nhật metrics
            processing_time = time.time() - start_time
            self._update_metrics(processing_time, success)
            
            # Thêm thông tin timing vào kết quả nếu là dict
            if result is not None and isinstance(result, dict):
                result["processing_time"] = processing_time
                result["success"] = success
                
            return result
    
    def _update_metrics(self, processing_time: float, success: bool):
        """
        Cập nhật metrics hiệu suất của agent.
        
        Args:
            processing_time: Thời gian xử lý (giây)
            success: Thành công hay thất bại
        """
        if not self.config.metrics_enabled:
            return
            
        self.metrics["total_requests"] += 1
        self.metrics["total_processing_time"] += processing_time
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
            
        self.metrics["avg_processing_time"] = (
            self.metrics["total_processing_time"] / self.metrics["total_requests"]
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Lấy metrics hiệu suất của agent.
        
        Returns:
            Dict chứa metrics
        """
        return self.metrics.copy()
    
    def export_metrics(self, output_path: str):
        """
        Xuất metrics ra file JSON.
        
        Args:
            output_path: Đường dẫn file output
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            self.logger.info(f"Metrics đã được xuất ra {output_path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi xuất metrics: {str(e)}")
    
    def remember(self, key: str, value: Any):
        """
        Lưu trữ thông tin trong bộ nhớ ngắn hạn.
        
        Args:
            key: Khóa để lưu trữ
            value: Giá trị cần lưu
        """
        if not self.config.memory_enabled:
            return
            
        self.short_term_memory[key] = {
            "value": value,
            "timestamp": time.time()
        }
        self.logger.debug(f"Đã lưu '{key}' vào bộ nhớ ngắn hạn")
    
    def recall(self, key: str) -> Optional[Any]:
        """
        Truy xuất thông tin từ bộ nhớ ngắn hạn.
        
        Args:
            key: Khóa cần truy xuất
            
        Returns:
            Giá trị đã lưu hoặc None nếu không tìm thấy
        """
        if not self.config.memory_enabled:
            return None
            
        memory_item = self.short_term_memory.get(key)
        if memory_item:
            return memory_item["value"]
        return None
    
    def clear_memory(self):
        """Xóa bộ nhớ ngắn hạn."""
        self.short_term_memory.clear()
        self.logger.debug("Đã xóa bộ nhớ ngắn hạn")
    
    def __str__(self) -> str:
        """String representation của agent."""
        status = "Đã khởi tạo" if self.initialized else "Chưa khởi tạo"
        return f"{self.name} Agent ({status})"