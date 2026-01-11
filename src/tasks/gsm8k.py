# GSM8K (Grade School Math 8K) task
from .base_task import BaseTask
import re


class GSM8K(BaseTask):
    def __init__(
        self,
        train_size,
        test_size,
        task_name: str,
        benchmark="gsm8k",
        task_description="Grade school math word problems",
        data_dir="",
        seed=None,
        **kwargs,
    ):
        self.options = {}
        self.benchmark = benchmark
        super().__init__(
            task_name=task_name,
            task_description=task_description,
            data_dir=data_dir,
            seed=seed,
            train_size=train_size,
            test_size=test_size,
            benchmark=benchmark,
            **kwargs,
        )

        self.task_name = task_name

    def _get_task_initial_prompt(self):
        base_prompt = "Solve this grade school math problem step by step."
        suffix = "<Question>{question}</Question>\nProvide your final numerical answer in <answer> and </answer> tags."
        initial_prompt = base_prompt + suffix
        return initial_prompt, base_prompt, suffix

    def clean_response(self, response):
        """
        Extract numerical answer from model response.
        GSM8K answers are numbers (possibly with commas, decimals, negatives).
        """
        # First try to find answer in tags
        clean_pattern = r"<answer>(.*?)<\/answer>"
        matches = re.findall(clean_pattern, response, re.IGNORECASE | re.DOTALL)

        if matches and matches[-1].strip():
            answer_text = matches[-1].strip()
        else:
            # Fallback: try to find the last number in the response
            answer_text = response.strip()

        # Extract number from the text
        # Handle formats like: "72", "$72", "72.5", "-72", "1,234"
        number_pattern = r'-?\$?\d+(?:,\d{3})*(?:\.\d+)?'
        numbers = re.findall(number_pattern, answer_text)

        if numbers:
            # Take the last number (usually the final answer)
            final_number = numbers[-1]
            # Remove $ and commas
            final_number = final_number.replace('$', '').replace(',', '')
            return final_number

        return "NO_ANSWER"
