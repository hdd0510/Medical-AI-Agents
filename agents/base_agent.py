from abc import ABC, abstractmethod
from typing import Dict, List, Any

import logging
import time
import json

class BaseAgent(ABC):
    """
    Base class for all agents with GPT-like reasoning interface.
    Each specialized agent inherits from this and implements their specific logic.
    """
    
    def __init__(self, 
                name: str,
                system_prompt: str,
                llm_client,
                tools: List[Tool] = None,
                memory_enabled: bool = True,
                device: str = "cuda"):
        
        self.name = name
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.tools = tools or []
        self.memory_enabled = memory_enabled
        self.device = device
        self.logger = logging.getLogger(f"agent.{self.name.lower().replace(' ', '_')}")
        self.cache = {}
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "tool_usages": {},
            "avg_processing_time": 0,
            "total_processing_time": 0
        }
    
    def register_tool(self, tool: Tool):
        """Register a new tool for this agent to use."""
        self.tools.append(tool)
        self.metrics["tool_usages"][tool.name] = 0
        
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request using agent's reasoning and tools.
        
        Args:
            input_data: Dictionary containing input including query, image paths, etc.
            
        Returns:
            Dictionary with processed results and metadata
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # 1. Create agent prompt with tools description
            full_prompt = self._create_agent_prompt(input_data)
            
            # 2. Get reasoning and tool calls from LLM
            reasoning_result = self._reason(full_prompt)
            
            # 3. Execute tool calls if any
            if "tool_calls" in reasoning_result:
                results = self._execute_tool_calls(reasoning_result["tool_calls"], input_data)
                reasoning_result["tool_results"] = results
            
            # 4. Get final answer from LLM using reasoning and tool results
            final_result = self._synthesize_answer(reasoning_result, input_data)
            
            # Record success and timing
            self.metrics["successful_requests"] += 1
            processing_time = time.time() - start_time
            self.metrics["total_processing_time"] += processing_time
            self.metrics["avg_processing_time"] = (
                self.metrics["total_processing_time"] / self.metrics["total_requests"]
            )
            
            # Add metadata to result
            final_result["success"] = True
            final_result["processing_time"] = processing_time
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            processing_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time
            }
    
    def _create_agent_prompt(self, input_data: Dict[str, Any]) -> str:
        """Create the prompt for the agent with context and tools."""
        # Start with system prompt
        prompt = self.system_prompt
        
        # Add tools information
        tools_description = "\n\nAvailable Tools:\n"
        for tool in self.tools:
            spec = tool.get_spec()
            tools_description += f"\n- {spec['name']}: {spec['description']}\n"
            tools_description += f"  Parameters: {json.dumps(spec['parameters'], indent=2)}\n"
            tools_description += f"  Returns: {json.dumps(spec['returns'], indent=2)}\n"
        
        prompt += tools_description
        
        # Add format instructions
        prompt += """
        \nResponse Format:
        {
            "reasoning": "Your step-by-step reasoning about the problem",
            "tool_calls": [
                {
                    "tool_name": "name_of_tool_to_use",
                    "parameters": {
                        "param1": "value1",
                        "param2": "value2"
                    }
                }
            ],
            "answer": "Your final answer based on reasoning and tool results"
        }
        
        If you don't need to use any tools, you can omit the "tool_calls" field.
        """
        
        # Add query/input specific information
        prompt += f"\n\nCurrent Request:\n{json.dumps(input_data, indent=2)}"
        
        return prompt
    
    def _reason(self, prompt: str) -> Dict[str, Any]:
        """
        Use LLM to reason about the problem and decide which tools to use.
        
        Args:
            prompt: The complete prompt with tools info and query
            
        Returns:
            Dictionary with reasoning and tool calls
        """
        # Call LLM with prompt
        response = self.llm_client.chat.completions.create(
            model="gpt-4",  # or any appropriate model
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2
        )
        
        # Extract content
        content = response.choices[0].message.content
        
        # Try to parse JSON output
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_content = content[start_idx:end_idx]
            return json.loads(json_content)
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            # Fallback: return just the reasoning and answer without tool calls
            return {
                "reasoning": content,
                "answer": content
            }
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool calls requested by the agent's reasoning.
        
        Args:
            tool_calls: List of tool calls with name and parameters
            input_data: Original input data for context
            
        Returns:
            Dictionary mapping tool names to their results
        """
        results = {}
        
        for call in tool_calls:
            tool_name = call["tool_name"]
            parameters = call["parameters"]
            
            # Find the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                results[tool_name] = {
                    "error": f"Tool '{tool_name}' not found",
                    "success": False
                }
                continue
            
            # Execute the tool
            try:
                # Add common parameters like image_path if needed
                if "image_path" in input_data and "image_path" not in parameters:
                    parameters["image_path"] = input_data["image_path"]
                    
                tool_result = tool.run(**parameters)
                results[tool_name] = tool_result
                
                # Update metrics
                self.metrics["tool_usages"][tool_name] = self.metrics["tool_usages"].get(tool_name, 0) + 1
                
            except Exception as e:
                self.logger.error(f"Error executing tool '{tool_name}': {str(e)}")
                results[tool_name] = {
                    "error": str(e),
                    "success": False
                }
        
        return results
    
    def _synthesize_answer(self, reasoning_result: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize final answer based on reasoning and tool results.
        
        Args:
            reasoning_result: The reasoning and tool results
            input_data: Original input data
            
        Returns:
            Final answer and metadata
        """
        # If there are no tool results, just return the reasoning answer
        if "tool_results" not in reasoning_result:
            return {
                "answer": reasoning_result.get("answer", ""),
                "reasoning": reasoning_result.get("reasoning", ""),
                "confidence": 0.9  # Default high confidence when no tools used
            }
        
        # Create a prompt for synthesizing the answer
        synthesis_prompt = f"""
        You previously analyzed this request:
        {json.dumps(input_data, indent=2)}
        
        Your reasoning was:
        {reasoning_result['reasoning']}
        
        You used tools and got these results:
        {json.dumps(reasoning_result['tool_results'], indent=2)}
        
        Please synthesize a final answer based on all this information.
        Response format:
        {{
            "answer": "Your comprehensive final answer",
            "confidence": 0.XX,  # Your confidence from 0.0 to 1.0
            "explanation": "Brief explanation of your confidence assessment"
        }}
        """
        
        # Call LLM for synthesis
        response = self.llm_client.chat.completions.create(
            model="gpt-4",  # or any appropriate model
            messages=[{"role": "system", "content": synthesis_prompt}],
            temperature=0.2
        )
        
        # Extract and parse content
        content = response.choices[0].message.content
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_content = content[start_idx:end_idx]
            synthesis = json.loads(json_content)
            
            # Create the final result with original reasoning and tool results
            final_result = {
                "answer": synthesis.get("answer", ""),
                "confidence": synthesis.get("confidence", 0.5),
                "reasoning": reasoning_result.get("reasoning", ""),
                "tool_results": reasoning_result.get("tool_results", {}),
                "explanation": synthesis.get("explanation", "")
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error parsing synthesis response: {e}")
            # Fallback
            return {
                "answer": reasoning_result.get("answer", content),
                "reasoning": reasoning_result.get("reasoning", ""),
                "tool_results": reasoning_result.get("tool_results", {}),
                "confidence": 0.5
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return self.metrics.copy()