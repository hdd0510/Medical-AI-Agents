# Medical AI System

Hệ thống AI y tế đa agent với các chức năng chuyên biệt cho xử lý hình ảnh y tế và trả lời câu hỏi y tế.

## Cấu trúc thư mục

```
medical-ai-system/
├── agents/                    # Các agent AI
│   ├── vqa/                  # Agent trả lời câu hỏi về hình ảnh
│   ├── detector/             # Agent phát hiện đối tượng y tế
│   ├── classifier_1/         # Agent phân loại hình ảnh 1
│   ├── classifier_2/         # Agent phân loại hình ảnh 2
│   └── rag/                  # Agent truy xuất và tạo câu trả lời
│
├── data/                     # Dữ liệu
│   ├── images/              # Hình ảnh y tế
│   ├── documents/           # Tài liệu y tế
│   └── embeddings/          # Vector embeddings
│
├── memory/                   # Quản lý bộ nhớ
│   ├── vector_store.py      # Lưu trữ vector
│   └── conversation_memory.py # Lưu trữ hội thoại
│
├── orchestrator/             # Điều phối các agent
│   ├── main.py              # Điểm vào chính
│   └── workflow.py          # Quy trình xử lý
│
├── tests/                    # Kiểm thử
│   ├── test_vqa.py
│   ├── test_detector.py
│   ├── test_rag.py
│   └── test_integration.py
│
└── tools/                    # Công cụ hỗ trợ
    ├── image_utils.py       # Xử lý hình ảnh
    ├── text_utils.py        # Xử lý văn bản
    ├── document_loader.py   # Đọc tài liệu
    └── web_scraper.py       # Thu thập dữ liệu web
```

## Cài đặt

1. Tạo môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. VQA Agent
```python
from agents.vqa import MedicalVQAAgent, VQAConfig

config = VQAConfig(model_path="path/to/model")
agent = MedicalVQAAgent(config)

result = agent.answer_question(
    image_path="path/to/image.jpg",
    question="Mô tả hình ảnh này"
)
print(result["answer"])
```

### 2. Detector Agent
```python
from agents.detector import MedicalDetectorAgent, DetectorConfig

config = DetectorConfig(model_path="path/to/model")
agent = MedicalDetectorAgent(config)

result = agent.detect("path/to/image.jpg")
print(result["detections"])
```

### 3. Classifier Agents
```python
from agents.classifier_1 import MedicalClassifierAgent1, ClassifierConfig1
from agents.classifier_2 import MedicalClassifierAgent2, ClassifierConfig2

# Classifier 1
config1 = ClassifierConfig1(model_path="path/to/model1")
agent1 = MedicalClassifierAgent1(config1)
result1 = agent1.classify("path/to/image.jpg")

# Classifier 2
config2 = ClassifierConfig2(model_path="path/to/model2")
agent2 = MedicalClassifierAgent2(config2)
result2 = agent2.classify("path/to/image.jpg")
```

### 4. RAG Agent
```python
from agents.rag import MedicalRAGAgent, RAGConfig

config = RAGConfig(
    llm_model_path="path/to/llm",
    embedding_model_path="path/to/embedding"
)
agent = MedicalRAGAgent(config)

result = agent.answer_question("Câu hỏi y tế của bạn")
print(result["answer"])
```

## Kiểm thử

Chạy các test:
```bash
python -m pytest tests/
```

## Đóng góp

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/amazing-feature`)
3. Commit thay đổi (`git commit -m 'Add some amazing feature'`)
4. Push lên branch (`git push origin feature/amazing-feature`)
5. Tạo Pull Request

## Giấy phép

Dự án này được cấp phép theo MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết. 