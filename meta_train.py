from src.runner import Runner
from dotenv import load_dotenv
import argparse
import os
import yaml


def load_config(args, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    args.meta_train_tasks = config["meta_train_tasks"]
    args.meta_test_tasks = []

    return args

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--init_system_prompt_path", type=str, default="./prompts/default.json")

    # Search Settings
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=3)
    parser.add_argument("--num_system_candidate", type=int, default=9)
    parser.add_argument("--num_user_candidate", type=int, default=3)
    parser.add_argument("--user_top_k", type=int, default=3)

    # Base Model Settings
    parser.add_argument("--base_model_type", type=str, required=True)
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--base_model_temperature", type=float, default=0.0)

    # Optimizer Model Settings
    parser.add_argument("--optim_model_type", type=str, default="openai")
    parser.add_argument("--optim_model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--optim_model_temperature", type=float, default=1.0)

    # Task Settings
    parser.add_argument("--task_config_path", type=str, default="./configs/amazon.yaml")
    parser.add_argument("--train_size", type=int, default=50)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_dir", type=str, default="./datasets")

    args = parser.parse_args()
    args = load_config(args, args.task_config_path)

    # Load API keys from .env file
    load_dotenv()
    args.openai_api_key = os.getenv("OPENAI_API_KEY")
    args.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    args.runpod_api_key = os.getenv("RUNPOD_API_KEY")
    args.runpod_endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")

    return args

if __name__ == "__main__":
    args = get_args()
    runner = Runner(args)
    print("✅ Runner initialized! Starting training...", flush=True)
    runner.meta_train()
    print("✅ Training complete!", flush=True)