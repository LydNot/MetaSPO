"""
Math task with LLM-as-judge evaluation
Author: Claude Opus 4.5 (with human guidance)
Enables MetaSPO on explanation-based math tasks where exact matching fails
"""
from .math import Math
import numpy as np
from openai import OpenAI
import os
import time


class MathLLMJudge(Math):
    """
    Math task that uses an LLM to judge if predictions are correct.
    This handles cases where exact string matching fails but answers are mathematically equivalent.
    """

    def __init__(self, *args, judge_model="gpt-4o-mini", judge_api_key=None, task_name=None, **kwargs):
        # Strip _judged suffix from task_name for data loading
        if task_name and task_name.endswith("_judged"):
            actual_task_name = task_name.replace("_judged", "")
            super().__init__(*args, task_name=actual_task_name, **kwargs)
            # But keep the full name for identification
            self.task_name = task_name
        else:
            super().__init__(*args, task_name=task_name, **kwargs)

        # Initialize judge model
        self.judge_model_name = judge_model
        if judge_api_key is None:
            judge_api_key = os.getenv("OPENAI_API_KEY")

        self.judge_client = OpenAI(api_key=judge_api_key)

        # Cache for judge decisions (avoid re-judging same pairs)
        self.judge_cache = {}

        print(f"✓ LLM-as-judge enabled: {judge_model}")

    def _call_judge(self, prediction, ground_truth, question):
        """
        Call LLM judge to determine if prediction is correct.

        Returns:
            1 if correct, 0 if incorrect
        """
        # Create cache key
        cache_key = (prediction[:100], ground_truth[:100], question[:100])
        if cache_key in self.judge_cache:
            return self.judge_cache[cache_key]

        judge_prompt = f"""You are an expert mathematics evaluator. Your task is to determine if a student's answer is mathematically correct.

**Question:**
{question}

**Ground Truth Reference Answer:**
{ground_truth}

**Student's Answer:**
{prediction}

**Instructions:**
1. Evaluate if the student's answer is mathematically correct and addresses the question
2. The student's explanation may differ from the reference but can still be correct
3. Focus on mathematical correctness, not stylistic similarity
4. Minor notational differences are acceptable
5. If the core mathematical reasoning and conclusion are correct, mark as CORRECT

**Output Format:**
Respond with ONLY one word:
- "CORRECT" if the student's answer is mathematically sound and addresses the question
- "INCORRECT" if the answer has mathematical errors or doesn't address the question

Your judgment:"""

        max_retries = 3
        retry_count = 0
        backoff_time = 1

        while retry_count < max_retries:
            try:
                response = self.judge_client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {"role": "system", "content": "You are an expert mathematics evaluator."},
                        {"role": "user", "content": judge_prompt}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )

                judgment = response.choices[0].message.content.strip().upper()

                # Parse judgment
                if "CORRECT" in judgment:
                    result = 1
                else:
                    result = 0

                # Cache the result
                self.judge_cache[cache_key] = result
                return result

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Judge API error after {max_retries} attempts: {e}")
                    # Default to incorrect if judge fails
                    return 0
                time.sleep(backoff_time)
                backoff_time *= 1.5

        return 0

    def set_current_questions(self, questions):
        """Store current batch questions for LLM judge."""
        self.current_batch_questions = questions

    def cal_correct(self, preds, labels):
        """
        Override to use LLM judge instead of exact string matching.
        """
        # Get the questions from the current batch
        questions = getattr(self, 'current_batch_questions', [""] * len(preds))

        correct_list = []
        for pred, label, question in zip(preds, labels, questions):
            # Use LLM judge to determine correctness
            is_correct = self._call_judge(pred, label, question)
            correct_list.append(is_correct)

        return correct_list

    def cal_metric(self, preds, labels):
        """
        Calculate accuracy using LLM judge.
        """
        correct = self.cal_correct(preds=preds, labels=labels)
        accuracy = np.mean(correct)

        # Print judge statistics
        num_correct = sum(correct)
        print(f"  [LLM Judge] {num_correct}/{len(correct)} correct ({accuracy*100:.1f}%)")

        return accuracy
