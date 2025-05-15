#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Command Line Interface
-----------------------------------------
Interface dòng lệnh cho hệ thống AI y tế đa agent.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("medical-ai-cli")

# Import orchestrator
try:
    from orchestrator.main import MedicalOrchestrator, OrchestratorConfig
except ImportError:
    logger.error("Không thể import Orchestrator. Vui lòng kiểm tra cài đặt.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Medical AI System - Hệ thống AI y tế đa agent"
    )
    
    # Các lệnh con
    subparsers = parser.add_subparsers(dest="command", help="Lệnh cần thực thi")
    
    # Lệnh analyze
    analyze_parser = subparsers.add_parser("analyze", help="Phân tích hình ảnh y tế")
    analyze_parser.add_argument(
        "--image", "-i", 
        type=str, 
        required=True,
        help="Đường dẫn đến hình ảnh cần phân tích"
    )
    analyze_parser.add_argument(
        "--query", "-q", 
        type=str, 
        default=None,
        help="Câu hỏi về hình ảnh (tùy chọn)"
    )
    analyze_parser.add_argument(
        "--medical-context", "-m", 
        type=str, 
        default=None,
        help="Đường dẫn đến file JSON chứa thông tin y tế bổ sung (tùy chọn)"
    )
    analyze_parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="results",
        help="Thư mục lưu kết quả (mặc định: results)"
    )
    analyze_parser.add_argument(
        "--device", "-d", 
        type=str, 
        choices=["cuda", "cpu"], 
        default="cuda",
        help="Thiết bị xử lý (mặc định: cuda)"
    )
    analyze_parser.add_argument(
        "--parallel", "-p", 
        action="store_true",
        help="Sử dụng xử lý song song"
    )
    analyze_parser.add_argument(
        "--no-reflection", "-nr", 
        action="store_true",
        help="Tắt cơ chế reflection"
    )
    
    # Lệnh serve
    serve_parser = subparsers.add_parser("serve", help="Khởi chạy API service")
    serve_parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host address (mặc định: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port (mặc định: 8000)"
    )
    serve_parser.add_argument(
        "--device", 
        type=str, 
        choices=["cuda", "cpu"], 
        default="cuda",
        help="Thiết bị xử lý (mặc định: cuda)"
    )
    
    # Lệnh version
    version_parser = subparsers.add_parser("version", help="Hiển thị phiên bản")
    
    return parser.parse_args()


def command_analyze(args):
    """Thực hiện lệnh phân tích hình ảnh."""
    # Kiểm tra file ảnh
    if not os.path.exists(args.image):
        logger.error(f"Không tìm thấy file ảnh: {args.image}")
        return 1
    
    # Đọc medical context nếu có
    medical_context = None
    if args.medical_context and os.path.exists(args.medical_context):
        try:
            with open(args.medical_context, 'r', encoding='utf-8') as f:
                medical_context = json.load(f)
            logger.info(f"Đã đọc thông tin medical context từ {args.medical_context}")
        except Exception as e:
            logger.warning(f"Không thể đọc medical context: {str(e)}")
    
    # Tạo config
    config = OrchestratorConfig(
        device=args.device,
        parallel_execution=args.parallel,
        use_reflection=not args.no_reflection,
        output_path=args.output
    )
    
    # Khởi tạo orchestrator
    try:
        logger.info("Khởi tạo Medical Orchestrator...")
        orchestrator = MedicalOrchestrator(config)
        
        # Đăng ký các agents
        orchestrator.register_agents()
        
        # Phân tích hình ảnh
        logger.info(f"Bắt đầu phân tích hình ảnh: {args.image}")
        result = orchestrator.orchestrate(
            image_path=args.image,
            query=args.query,
            medical_context=medical_context
        )
        
        # In kết quả ra console
        if "error" in result:
            logger.error(f"Lỗi trong quá trình phân tích: {result['error']}")
            return 1
        
        # In thông tin chính
        print("\n=== KẾT QUẢ PHÂN TÍCH ===")
        if "answer" in result:
            print(f"\nCâu trả lời: {result['answer']}")
        
        if "detection_count" in result:
            print(f"\nSố polyp phát hiện: {result['detection_count']}")
        
        if "modality" in result:
            print(f"\nKỹ thuật nội soi: {result.get('modality', {}).get('class_name', 'Không xác định')}")
        
        if "region" in result:
            print(f"\nVị trí giải phẫu: {result.get('region', {}).get('class_name', 'Không xác định')}")
        
        if "summary" in result:
            print(f"\nTóm tắt: {result['summary']}")
        
        # Hiện đường dẫn đến file kết quả chi tiết
        session_id = result.get("metadata", {}).get("session_id", "unknown")
        result_path = os.path.join(args.output, session_id, "result.json")
        print(f"\nKết quả chi tiết đã được lưu tại: {result_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Lỗi không mong muốn: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def command_serve(args):
    """Khởi chạy API service."""
    try:
        import uvicorn
        from api.main import app, initialize_app
        
        # Khởi tạo app
        logger.info("Khởi tạo API service...")
        initialize_app(device=args.device)
        
        # Chạy server
        logger.info(f"Khởi động API server tại {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        
        return 0
        
    except ImportError:
        logger.error("Không thể import fastapi hoặc uvicorn. Vui lòng cài đặt: pip install fastapi uvicorn")
        return 1
    except Exception as e:
        logger.error(f"Lỗi khởi động API service: {str(e)}")
        return 1


def command_version(_):
    """Hiển thị phiên bản."""
    # Hardcoded version
    print("Medical AI System v0.1.0")
    return 0


def main():
    """Entry point cho chương trình."""
    # Parse arguments
    args = parse_args()
    
    # Nếu không có lệnh nào được chỉ định
    if not args.command:
        print("Vui lòng chỉ định một lệnh. Sử dụng --help để xem các lệnh khả dụng.")
        return 1
    
    # Thực thi lệnh tương ứng
    if args.command == "analyze":
        return command_analyze(args)
    elif args.command == "serve":
        return command_serve(args)
    elif args.command == "version":
        return command_version(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 