#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reflexion Module Initialization
------------------------------
Module initialization cho Reflexion system trong Medical AI.
"""

from .config import (
    ExecutionPlan, ExecutionStep, StepResult, CriticFeedback,
    ReflexionSession, ReflexionStage, ReflexionConfig, ReflexionMetrics,
    AgentType, StepStatus, CriticScore
)
from .planner import PlannerAgent
from .critic import CriticAgent

__all__ = [
    # Types
    'ExecutionPlan',
    'ExecutionStep',  
    'StepResult',
    'CriticFeedback',
    'ReflexionSession',
    'ReflexionStage',
    'ReflexionConfig',
    'ReflexionMetrics',
    'AgentType',
    'StepStatus',
    'CriticScore',
    
    # Agents
    'PlannerAgent',
    'CriticAgent'
]

# Version information
__version__ = "1.0.0"
__author__ = "Medical AI Team"
__email__ = "medical-ai@example.com"

# Module configuration
DEFAULT_CONFIG = {
    "max_iterations": 3,
    "min_score_threshold": 3.0,
    "step_timeout": 60.0,
    "total_timeout": 300.0,
    "enable_parallel_execution": True,
    "planner_model": "gpt-4",
    "critic_model": "gpt-4",
    "logging_level": "INFO"
}

# Quick factory functions
def create_default_config(**kwargs) -> ReflexionConfig:
    """Tạo ReflexionConfig với default values."""
    config_dict = {**DEFAULT_CONFIG, **kwargs}
    return ReflexionConfig(**config_dict)

def create_simple_plan(query: str, image_path: str, agent_types: list = None) -> ExecutionPlan:
    """Tạo ExecutionPlan đơn giản cho testing."""
    if agent_types is None:
        agent_types = [AgentType.DETECTOR, AgentType.CLASSIFIER_1, AgentType.VQA]
    
    plan = ExecutionPlan(query=query, image_path=image_path)
    
    for i, agent_type in enumerate(agent_types):
        step = ExecutionStep(
            agent_type=agent_type,
            params={"query": query} if agent_type == AgentType.VQA else {}
        )
        plan.steps.append(step)
    
    return plan