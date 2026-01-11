"""
Claude model integration for MetaSPO
Author: Claude Opus 4.5 (with human guidance)
"""
from anthropic import Anthropic
import time

MODEL_DICT = {
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250514",
    "claude-sonnet-3.5": "claude-3-5-sonnet-20241022",
}


class ClaudeModel:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float,
        batch_mode: bool = True,
        **kwargs,
    ):
        if api_key is None:
            raise ValueError(f"api_key error: {api_key}")
        try:
            self.client = Anthropic(api_key=api_key)
        except Exception as e:
            print(f"Init Anthropic client error: \n{e}")
            raise RuntimeError("Failed to initialize Anthropic client") from e

        if model_name not in MODEL_DICT:
            raise ValueError(f"Model {model_name} not supported. Available: {list(MODEL_DICT.keys())}")

        self.model_name = MODEL_DICT[model_name]
        self.temperature = temperature
        self.batch_mode = batch_mode
        self.batch_forward_func = self.batch_forward_chatcompletion
        self.generate = self.claude_chat_completion

    def batch_forward_chatcompletion(self, batch_prompts):
        return [self.claude_chat_completion(prompt=prompt) for prompt in batch_prompts]

    def claude_chat_completion(self, prompt):
        """
        Send a chat completion request to Claude API.

        Args:
            prompt: List of messages in format [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
                   or [{"role": "user", "content": "..."}] without system message

        Returns:
            str: The model's response content
        """
        backoff_time = 1
        max_retries = 5
        retry_count = 0

        # Convert OpenAI-style messages to Claude format
        # Claude API requires system message separate from messages list
        system_message = None
        messages = []

        for msg in prompt:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        while retry_count < max_retries:
            try:
                # Build request kwargs
                request_kwargs = {
                    "model": self.model_name,
                    "max_tokens": 4096,  # Claude requires explicit max_tokens
                    "temperature": self.temperature,
                    "messages": messages,
                }

                # Add system message if present
                if system_message:
                    request_kwargs["system"] = system_message

                response = self.client.messages.create(**request_kwargs)

                # Extract text from response
                # Claude returns response.content as a list of content blocks
                return response.content[0].text.strip()

            except Exception as e:
                retry_count += 1
                error_msg = f"Claude API error (attempt {retry_count}/{max_retries}): {e}"
                print(error_msg)

                if retry_count >= max_retries:
                    print(f"Max retries reached. Failing.")
                    raise RuntimeError(f"Claude API failed after {max_retries} attempts: {e}") from e

                print(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 1.5
