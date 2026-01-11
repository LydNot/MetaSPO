# define task prompts for math datasets
from .base_task import BaseTask
import re


class Math(BaseTask):
    def __init__(
        self,
        train_size,
        test_size,
        task_name: str,
        benchmark="math",
        task_description="Mathematics learning benchmark",
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
        base_prompt = "Help solve this mathematics problem with clear explanation."
        suffix = "<Question>{question}</Question>\nProvide your answer with reasoning in <answer> and </answer> tags."
        initial_prompt = base_prompt + suffix
        return initial_prompt, base_prompt, suffix

    def clean_response(self, response):
        """
        Extract answer from model response.
        StackMathQA answers are text-based, so we extract content from <answer> tags.
        """
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"
        
        matches = re.findall(clean_pattern, response, re.IGNORECASE | re.DOTALL)
        
        # No answer tag found
        if not matches or not matches[-1].strip():
            # Fallback: return the whole response
            return response.strip() if response.strip() else "NO_ANSWER"
        
        # Return the content inside answer tags
        return matches[-1].strip()
