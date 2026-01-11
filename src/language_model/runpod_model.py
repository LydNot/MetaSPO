"""
Runpod Model - Client for Runpod serverless vLLM endpoints.

Supports models like Llama, Deepseek, Qwen, and other models hosted on Runpod.
Uses Runpod's native serverless API format.

Environment variables:
    RUNPOD_API_KEY: Your Runpod API key
    RUNPOD_ENDPOINT_ID: Your serverless endpoint ID (or pass via config)
"""

import requests
import time
import os
from typing import List, Optional


# Common Runpod-hosted models and their typical identifiers
MODEL_DICT = {
    # Llama models
    "llama3.1_8B": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.1_70B": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "llama3.2_3B": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.3_70B": "meta-llama/Llama-3.3-70B-Instruct",
    # Deepseek models
    "deepseek_7B": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek_v2": "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek_v2.5": "deepseek-ai/DeepSeek-V2.5",
    "deepseek_v3": "deepseek-ai/DeepSeek-V3",
    "deepseek_r1": "deepseek-ai/DeepSeek-R1",
    # Qwen models
    "qwen2.5_7B": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5_72B": "Qwen/Qwen2.5-72B-Instruct",
    # Qwen3 models
    "qwen3_1.7B": "Qwen/Qwen3-1.7B",
    "qwen3_4B": "Qwen/Qwen3-4B",
    "qwen3_8B": "Qwen/Qwen3-8B",
    "qwen3_14B": "Qwen/Qwen3-14B",
    "qwen3_32B": "Qwen/Qwen3-32B",
}


class RunpodModel:
    """
    Runpod serverless model client using native Runpod API.
    
    Args:
        model_name: Model identifier (key from MODEL_DICT or full HuggingFace path)
        api_key: Runpod API key (or set RUNPOD_API_KEY env var)
        endpoint_id: Runpod serverless endpoint ID (or set RUNPOD_ENDPOINT_ID env var)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        batch_mode: Whether to use batch processing (default True)
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        runpod_api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        batch_mode: bool = True,
        **kwargs,
    ):
        # Resolve API key (accept either api_key or runpod_api_key)
        self.api_key = runpod_api_key or api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Runpod API key required. Pass 'api_key' or set RUNPOD_API_KEY env var."
            )
        
        # Resolve endpoint ID
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID")
        if not self.endpoint_id:
            raise ValueError(
                "Runpod endpoint ID required. Pass 'endpoint_id' or set RUNPOD_ENDPOINT_ID env var."
            )
        
        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"
        
        # Resolve model name
        self.model_name = MODEL_DICT.get(model_name, model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_mode = batch_mode
        
        # Set batch forward function
        self.batch_forward_func = self._batch_forward
        self.generate = self._chat_completion

    def _format_prompt(self, messages: List[dict]) -> str:
        """Convert chat messages to a single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    def _batch_forward(self, batch_prompts: List[List[dict]]) -> List[str]:
        """Process a batch of prompts sequentially."""
        return [self._chat_completion(prompt) for prompt in batch_prompts]

    def _chat_completion(self, prompt: List[dict], max_retries: int = 5) -> str:
        """
        Send a completion request using Runpod's native API.
        
        Args:
            prompt: List of message dicts with 'role' and 'content' keys
            max_retries: Maximum number of retry attempts
            
        Returns:
            Model response text
        """
        formatted_prompt = self._format_prompt(prompt)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "input": {
                "prompt": formatted_prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        }
        
        backoff_time = 1
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Use runsync for synchronous requests
                response = requests.post(
                    f"{self.base_url}/runsync",
                    headers=headers,
                    json=payload,
                    timeout=300,  # 5 minute timeout for model inference
                )
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                data = response.json()
                
                if data.get("status") == "COMPLETED":
                    # Extract response from vLLM output format
                    output = data.get("output", [])
                    if output and isinstance(output, list):
                        choices = output[0].get("choices", [])
                        if choices:
                            tokens = choices[0].get("tokens", [])
                            if tokens:
                                return tokens[0].strip() if isinstance(tokens, list) else str(tokens).strip()
                    # Fallback: try direct output
                    if isinstance(output, str):
                        return output.strip()
                    raise Exception(f"Unexpected output format: {data}")
                
                elif data.get("status") == "FAILED":
                    raise Exception(f"Job failed: {data.get('error', 'Unknown error')}")
                
                else:
                    raise Exception(f"Unexpected status: {data.get('status')}")
                
            except Exception as e:
                last_error = e
                print(f"Runpod API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {backoff_time:.1f}s...")
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 1.5, 60)
        
        raise RuntimeError(
            f"Runpod API failed after {max_retries} attempts. Last error: {last_error}"
        )
