import importlib
from .base_task import BaseTask

# Add your new benchmark here with task names.
MEDMCQA_TASKS = [
    "anatomy",
    "surgery",
    "ob_gyn",
    "medicine",
    "pharmacology",
    "dental",
    "pediatrics",
    "pathology",
]
AMAZON_TASKS = ["beauty", "game", "baby", "office", "sports", "electronics", "pet"]
BIGBENCH_TASKS = [
    "logic_grid_puzzle",
    "logical_deduction",
    "temporal_sequences",
    "tracking_shuffled_objects",
    "object_counting",
    "reasoning_colored_objects",
    "epistemic",
    "navigate",
]
SAFETY_TASKS = [
    "ethos",
    "liar",
    "hatecheck",
    "sarcasm",
    "tweet_eval",
    "antropic_harmless",
]
GROUNDING_TASKS = [
    "hotpot_qa",
    "natural_questions",
    "squad",
    "web_qa",
    "drop",
    "trivia_qa",
]

MATH_TASKS = [
    "algebra",
    "geometry",
    "calculus",
    "probability",
    "algebra_small",
    "geometry_small",
    "calculus_small",
    "probability_small",
]

# Math tasks with LLM-as-judge evaluation (append _judged)
MATH_JUDGED_TASKS = [
    "algebra_judged",
    "geometry_judged",
    "calculus_judged",
    "probability_judged",
    "algebra_small_judged",
    "geometry_small_judged",
    "calculus_small_judged",
    "probability_small_judged",
]

# Open-ended tasks (no fixed ground truth)
OPEN_ENDED_TASKS = [
    "reasoning",
    "creative_writing",
]


def get_task(task_name):
    # Handle LLM-judged math tasks
    if task_name in MATH_JUDGED_TASKS:
        class_name = "MathLLMJudge"
        # Remove _judged suffix for data loading
        actual_task_name = task_name.replace("_judged", "")
    elif task_name in GROUNDING_TASKS:
        class_name = "Grounding"
    elif task_name in SAFETY_TASKS:
        class_name = "Safety"
    elif task_name in BIGBENCH_TASKS:
        class_name = "Bigbench"
    elif task_name in MEDMCQA_TASKS:
        class_name = "MEDMCQA"
    elif task_name in AMAZON_TASKS:
        class_name = "Amazon"
    elif task_name in MATH_TASKS:
        class_name = "Math"
    elif task_name in OPEN_ENDED_TASKS:
        if task_name == "reasoning":
            class_name = "OpenEndedReasoningTask"
        elif task_name == "creative_writing":
            class_name = "CreativeWritingTask"
        else:
            class_name = "OpenEndedTask"
    else:
        raise ValueError(f"{task_name} is not a recognized task")

    try:
        # Handle open-ended tasks specially
        if task_name in OPEN_ENDED_TASKS:
            module = importlib.import_module(".open_ended_task", package=__package__)
        elif task_name in MATH_JUDGED_TASKS:
            module = importlib.import_module(".math_llm_judge", package=__package__)
        else:
            module = importlib.import_module(f".{class_name.lower()}", package=__package__)
        CustomTask = getattr(module, class_name)

    except ModuleNotFoundError:
        raise ValueError(f"Module for task '{task_name}' could not be found.")

    return CustomTask
