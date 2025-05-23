# Medical AI System - Hệ thống AI y tế đa agent

Hệ thống AI y tế đa agent tích hợp nhiều mô hình AI để phân tích hình ảnh nội soi tiêu hóa, hỗ trợ phát hiện và phân loại polyp, đồng thời trả lời các câu hỏi y tế với cơ chế reflection để giảm thiểu bias.

## Tổng quan

Hệ thống được thiết kế theo kiến trúc đa agent, trong đó mỗi agent đảm nhận một nhiệm vụ chuyên biệt:

- **Detector Agent**: Phát hiện vị trí của polyp trong hình ảnh nội soi
- **Classifier Agent 1**: Phân loại kỹ thuật chụp ảnh nội soi (WLI, BLI, LCI)
- **Classifier Agent 2**: Phân loại vị trí giải phẫu trong đường tiêu hóa
- **VQA Agent**: Trả lời câu hỏi dựa trên phân tích hình ảnh
- **RAG Agent**: Tích hợp thông tin từ tài liệu y khoa để bổ sung cho câu trả lời

Tất cả các agent được điều phối bởi **Medical Orchestrator**, thành phần trung tâm chịu trách nhiệm lập kế hoạch, điều phối các agent, và tổng hợp kết quả.

## Tính năng chính

- **Phát hiện polyp** trong hình ảnh nội soi tiêu hóa
- **Phân loại kỹ thuật chụp** (WLI, BLI, LCI) và **vị trí giải phẫu**
- **Trả lời câu hỏi y tế** dựa trên hình ảnh
- **Phản biện và giảm thiểu bias** thông qua cơ chế reflection
- **Tích hợp kiến thức y khoa** từ nhiều nguồn
- **Xử lý song song** cho tối ưu hiệu suất
- **Lưu trữ và truy xuất** lịch sử phân tích

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                   Medical Orchestrator                   │
└────────────┬────────────┬────────────┬────────────┬─────┘
             │            │            │            │
┌────────────▼─┐ ┌────────▼───────┐ ┌──▼──────┐ ┌───▼─────┐
│    Detector   │ │   Classifier   │ │   VQA   │ │   RAG   │
│     Agent     │ │     Agents     │ │  Agent  │ │  Agent  │
└──────────────┘ └────────────────┘ └─────────┘ └─────────┘
       │                │                │            │
       └────────────────┴────────────────┴────────────┘
                               │
                     ┌─────────▼─────────┐
                     │  Memory System &  │
                     │  Vector Database  │
                     └───────────────────┘
```

## Cách sử dụng

### Cài đặt

```bash
# Clone repository
git clone https://github.com/yourusername/medical-ai-system.git
cd medical-ai-system

# Tạo môi trường virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt các dependencies
pip install -r requirements.txt

# Tải pre-trained models (nếu có)
python scripts/download_models.py
```

### Sử dụng cơ bản

```python
from orchestrator.main import MedicalOrchestrator, OrchestratorConfig

# Tạo config
config = OrchestratorConfig(
    device="cuda",  # hoặc "cpu"
    parallel_execution=True,
    use_reflection=True
)

# Khởi tạo orchestrator
orchestrator = MedicalOrchestrator(config)

# Đăng ký các agents (có thể tự động hoặc thủ công)
orchestrator.register_agents()

# Phân tích hình ảnh
result = orchestrator.orchestrate(
    image_path="path/to/image.jpg",
    query="Is there a polyp in this image?",
    medical_context={"patient_history": "Family history of colon cancer"}
)

# In kết quả
print(result["answer"])
print(f"Confidence: {result['confidence']}")
```

### API Service

Bạn cũng có thể chạy hệ thống như một API service:

```bash
# Khởi động API server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Sau đó gửi requests đến API:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/image.jpg" \
  -F "query=Is there a polyp in this image?"
```

## Testing

Hệ thống đi kèm với bộ test toàn diện:

```bash
# Chạy tất cả tests
pytest tests/

# Chạy test cụ thể
pytest tests/test_agents.py
pytest tests/test_orchestrator.py
pytest tests/test_integration.py
```

## Yêu cầu hệ thống

- Python 3.8+
- CUDA 11.7+ (cho GPU acceleration)
- 16GB RAM (tối thiểu)
- 8GB VRAM (cho CUDA acceleration)

## License

[MIT License](LICENSE)

## Citation

Nếu bạn sử dụng hệ thống này trong công trình nghiên cứu, vui lòng trích dẫn:

```
@software{medical_ai_system,
  author = {Your Name},
  title = {Medical AI System - Multi-agent AI System for Medical Image Analysis},
  year = {2023},
  url = {https://github.com/yourusername/medical-ai-system}
}
``` 