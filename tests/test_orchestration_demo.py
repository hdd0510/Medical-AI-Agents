#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Orchestration Demo for Medical AI System
--------------------------------------------
Demo cách sử dụng Medical Orchestrator với các agent thực.
"""

import os
import sys
import unittest
import pytest
import tempfile
import numpy as np
from PIL import Image

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import orchestrator và các agent
from orchestrator.main import MedicalOrchestrator, OrchestratorConfig
from agents.detector import DetectorAgent, DetectorAgentConfig
from agents.classifier_1 import MedicalClassifierAgent1, ClassifierConfig1
from agents.classifier_2 import MedicalClassifierAgent2, ClassifierConfig2
from agents.vqa import MedicalVQAAgent, VQAAgentConfig

# Cấu hình đường dẫn đến weights
WEIGHTS_DIR = "weights"
DETECTOR_WEIGHTS = os.path.join(WEIGHTS_DIR, "detect_best.pt")
CLASSIFIER1_WEIGHTS = os.path.join(WEIGHTS_DIR, "modal_best.pt")
CLASSIFIER2_WEIGHTS = os.path.join(WEIGHTS_DIR, "location_best.pt")
VQA_MODEL_WEIGHTS = os.path.join(WEIGHTS_DIR, "llava-med-mistral-v1.5-7b")

# Tạo ảnh test giả lập
def create_test_image(width=224, height=224):
    """Tạo ảnh test cho việc kiểm thử."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img_pil = Image.fromarray(img)
    return img_pil


def test_orchestration_demo():
    """Demo cách thiết lập và sử dụng Medical Orchestrator."""
    
    # Sửa import RAG trong orchestrator/main.py tạm thời
    import importlib.util
    import types
    
    # Tạo môđun giả để mock RAG agent
    mock_module = types.ModuleType('agents.rag')
    mock_module.MedicalRAGAgent = type('MedicalRAGAgent', (), {})
    mock_module.RAGConfig = type('RAGConfig', (), {})
    sys.modules['agents.rag'] = mock_module
    
    # 1. Tạo Orchestrator config
    orch_config = OrchestratorConfig(
        name="Demo Orchestrator",
        device="cpu",  # Sử dụng CPU để test
        parallel_execution=True,  # Bật xử lý song song
        use_reflection=False,     # Tắt reflection để đơn giản hóa
        memory_enabled=False      # Tắt memory để đơn giản hóa
    )
    
    # 2. Khởi tạo Orchestrator
    print("Khởi tạo Orchestrator...")
    orchestrator = MedicalOrchestrator(orch_config)
    
    # 3. Khởi tạo và đăng ký các agents
    print("Khởi tạo các agents...")
    
    # Detector Agent
    detector_config = DetectorAgentConfig(
        name="DetectorAgent",
        model_path=DETECTOR_WEIGHTS,
        device="cpu",
        confidence_threshold=0.3
    )
    detector_agent = DetectorAgent(detector_config)
    orchestrator.register_agent("detector", detector_agent)
    
    # Modality Classifier Agent
    classifier1_config = ClassifierConfig1(
        name="ModalityClassifier",
        model_path=CLASSIFIER1_WEIGHTS,
        device="cpu"
    )
    classifier1_agent = MedicalClassifierAgent1(classifier1_config)
    orchestrator.register_agent("classifier_1", classifier1_agent)
    
    # Location Classifier Agent
    classifier2_config = ClassifierConfig2(
        name="LocationClassifier",
        model_path=CLASSIFIER2_WEIGHTS,
        device="cpu"
    )
    classifier2_agent = MedicalClassifierAgent2(classifier2_config)
    orchestrator.register_agent("classifier_2", classifier2_agent)
    
    # VQA Agent (chỉ khởi tạo nếu weights có sẵn)
    if os.path.exists(VQA_MODEL_WEIGHTS):
        vqa_config = VQAAgentConfig(
            name="VQAAgent",
            model_path=VQA_MODEL_WEIGHTS,
            device="cpu"
        )
        vqa_agent = MedicalVQAAgent(vqa_config)
        orchestrator.register_agent("vqa", vqa_agent)
    
    # 4. Tạo ảnh test
    print("Tạo ảnh test...")
    test_img = create_test_image()
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        test_img_path = tmp.name
        test_img.save(test_img_path)
        print(f"Lưu ảnh test tại: {test_img_path}")
    
    try:
        # 5. Thực hiện phân tích toàn diện
        print("\n--- Thực hiện phân tích toàn diện ---")
        result = orchestrator.orchestrate(
            image_path=test_img_path,
            query=None,  # Không có câu hỏi cụ thể
            medical_context=None  # Không có thông tin y tế bổ sung
        )
        
        # 6. In kết quả phân tích toàn diện
        print("\n=== Kết quả phân tích toàn diện ===")
        print(f"Session ID: {result.get('metadata', {}).get('session_id', 'N/A')}")
        print(f"Task type: {result.get('metadata', {}).get('task_type', 'N/A')}")
        print(f"Processing time: {result.get('metadata', {}).get('processing_time', 0):.2f}s")
        
        # 7. In kết quả phát hiện polyp
        if "detector" in result:
            detector_result = result["detector"]
            print("\n>> Kết quả phát hiện polyp:")
            print(f"   Thành công: {detector_result.get('success', False)}")
            print(f"   Số lượng đối tượng: {detector_result.get('detection_count', 0)}")
            
            if detector_result.get('success', False) and detector_result.get('detections'):
                for i, detection in enumerate(detector_result['detections']):
                    print(f"   Polyp {i+1}: {detection.get('class_name')} - Độ tin cậy: {detection.get('confidence', 0):.2f}")
        
        # 8. In kết quả phân loại kỹ thuật nội soi
        if "classifier_1" in result:
            modal_result = result["classifier_1"]
            print("\n>> Kết quả phân loại kỹ thuật nội soi:")
            print(f"   Thành công: {modal_result.get('success', False)}")
            if modal_result.get('success', False):
                print(f"   Kỹ thuật: {modal_result.get('class_name', 'Unknown')}")
                print(f"   Độ tin cậy: {modal_result.get('confidence', 0):.2f}")
        
        # 9. In kết quả phân loại vị trí
        if "classifier_2" in result:
            location_result = result["classifier_2"]
            print("\n>> Kết quả phân loại vị trí:")
            print(f"   Thành công: {location_result.get('success', False)}")
            if location_result.get('success', False):
                print(f"   Vị trí: {location_result.get('class_name', 'Unknown')}")
                print(f"   Độ tin cậy: {location_result.get('confidence', 0):.2f}")
        
        # 10. Thực hiện hỏi đáp hình ảnh nếu VQA agent đã được đăng ký
        if "vqa" in orchestrator.agents:
            print("\n--- Thực hiện VQA ---")
            vqa_query = "Mô tả những gì bạn thấy trong hình ảnh nội soi này?"
            
            vqa_result = orchestrator.orchestrate(
                image_path=test_img_path,
                query=vqa_query,
                medical_context=None
            )
            
            # 11. In kết quả VQA
            print("\n=== Kết quả VQA ===")
            print(f"Câu hỏi: {vqa_query}")
            if "vqa" in vqa_result:
                print(f"Câu trả lời: {vqa_result['vqa'].get('answer', 'Không có câu trả lời')}")
                print(f"Độ tin cậy: {vqa_result['vqa'].get('confidence', 0):.2f}")
        
        print("\nDemo hoàn tất!")
        return True
        
    except Exception as e:
        print(f"Lỗi trong quá trình demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Xóa file ảnh tạm
        try:
            os.unlink(test_img_path)
        except:
            pass


if __name__ == "__main__":
    result = test_orchestration_demo()
    sys.exit(0 if result else 1) 