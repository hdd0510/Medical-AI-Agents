#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reflexion Data Types and Enums
------------------------------
Định nghĩa các kiểu dữ liệu cho hệ thống Reflexion trong Medical AI System.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import time
import uuid
from datetime import datetime

class AgentType(Enum):
    """Các loại agent trong hệ thống."""
    DETECTOR = "detector"
    CLASSIFIER_1 = "classifier_1"  # Modality classifier
    CLASSIFIER_2 = "classifier_2"  # Region classifier
    VQA = "vqa"
    RAG = "rag"
    
class StepStatus(Enum):
    """Trạng thái thực thi của một bước."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"

class CriticScore(Enum):
    """Điểm đánh giá từ critic."""
    EXCELLENT = 5  # Kết quả hoàn hảo, không cần cải thiện
    GOOD = 4       # Kết quả tốt, có thể cải thiện nhỏ
    ACCEPTABLE = 3 # Kết quả chấp nhận được
    POOR = 2       # Kết quả kém, cần cải thiện
    FAILURE = 1    # Thất bại hoàn toàn, cần thực hiện lại

class ReflexionStage(Enum):
    """Các giai đoạn trong chu trình Reflexion."""
    PLANNING = "planning"
    EXECUTION = "execution"
    CRITICISM = "criticism"
    REFINEMENT = "refinement"
    COMPLETED = "completed"

@dataclass
class ExecutionStep:
    """Một bước thực thi trong kế hoạch."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.DETECTOR
    params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 60.0
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if isinstance(self.agent_type, str):
            self.agent_type = AgentType(self.agent_type)
        if isinstance(self.status, str):
            self.status = StepStatus(self.status)

@dataclass
class StepResult:
    """Kết quả của một bước thực thi."""
    step_id: str
    status: StepStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    processing_time: float = 0.0
    memory_usage: float = 0.0
    executed_at: float = field(default_factory=time.time)
    agent_name: str = ""
    
    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = StepStatus(self.status)

@dataclass
class ExecutionPlan:
    """Kế hoạch thực thi từ Planner."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    image_path: str = ""
    steps: List[ExecutionStep] = field(default_factory=list)
    reasoning: str = ""
    estimated_time: float = 0.0
    priority_level: int = 5  # 1-10, 10 là cao nhất
    created_at: float = field(default_factory=time.time)
    version: int = 1
    
@dataclass
class CriticFeedback:
    """Phản hồi từ Critic về kết quả thực thi."""
    plan_id: str
    overall_score: CriticScore
    step_scores: Dict[str, CriticScore] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    refinement_plan: Optional[ExecutionPlan] = None
    reasoning: str = ""
    confidence: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if isinstance(self.overall_score, str):
            self.overall_score = CriticScore(self.overall_score)

@dataclass
class ReflexionSession:
    """Một phiên Reflexion hoàn chất."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    image_path: str = ""
    stage: ReflexionStage = ReflexionStage.PLANNING
    plans: List[ExecutionPlan] = field(default_factory=list)
    results: List[StepResult] = field(default_factory=list)
    feedbacks: List[CriticFeedback] = field(default_factory=list)
    current_plan: Optional[ExecutionPlan] = None
    iteration_count: int = 0
    max_iterations: int = 3
    success: bool = False
    final_answer: str = ""
    total_time: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        if isinstance(self.stage, str):
            self.stage = ReflexionStage(self.stage)

@dataclass
class ReflexionConfig:
    """Cấu hình cho hệ thống Reflexion."""
    max_iterations: int = 3
    min_score_threshold: float = 3.0  # Điểm tối thiểu để chấp nhận kết quả
    step_timeout: float = 60.0
    total_timeout: float = 300.0
    enable_parallel_execution: bool = True
    retry_failed_steps: bool = True
    planner_model: str = "gpt-4"
    critic_model: str = "gpt-4"
    planner_temperature: float = 0.7
    critic_temperature: float = 0.1
    logging_level: str = "INFO"
    
@dataclass
class ReflexionMetrics:
    """Metrics cho việc đánh giá hiệu suất Reflexion."""
    session_id: str
    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    average_iteration_time: float = 0.0
    total_api_calls: int = 0
    planner_calls: int = 0
    critic_calls: int = 0
    final_score: Optional[CriticScore] = None
    improvement_ratio: float = 0.0  # Tỷ lệ cải thiện qua các iteration
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.final_score and isinstance(self.final_score, str):
            self.final_score = CriticScore(self.final_score)