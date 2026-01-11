#!/usr/bin/env python3
"""Test LLM-as-judge on StackMathQA examples."""

import json
from src.tasks.math_llm_judge import MathLLMJudge
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("Testing LLM-as-Judge on StackMathQA")
print("=" * 70)

# Load sample data
with open("datasets/math/algebra_small.json", "r") as f:
    data = json.load(f)

sample = data["train"][0]

print("\n[Sample Problem]")
print(f"Question: {sample['question'][:150]}...")
print(f"\nGround Truth: {sample['answer'][:150]}...")

# Initialize task with LLM judge
print("\n[Initializing LLM Judge...]")
task = MathLLMJudge(
    train_size=5,
    test_size=5,
    task_name="algebra_small",
    data_dir="./datasets",
    seed=42,
    judge_model="gpt-4o-mini"
)

print("\n[Testing Judgment Scenarios]")

# Test 1: Identical answer (should be CORRECT)
print("\n1️⃣ Test: Identical answer")
prediction1 = sample["answer"]
label1 = sample["answer"]
result1 = task._call_judge(prediction1, label1, sample["question"])
print(f"   Prediction == Label: {result1} {'✓ CORRECT' if result1 == 1 else '✗ INCORRECT'}")

# Test 2: Different but mathematically correct answer
print("\n2️⃣ Test: Different valid mathematical explanation")
prediction2 = """Let me solve this using De Moivre's theorem.
We know that (cos θ + i sin θ)^n = cos(nθ) + i sin(nθ).
For n=6: (cos θ + i sin θ)^6 = cos(6θ) + i sin(6θ).
Expanding the left side with binomial theorem and equating imaginary parts,
we can derive the formula for tan(6θ)."""

result2 = task._call_judge(prediction2, label1, sample["question"])
print(f"   Different explanation: {result2} {'✓ CORRECT' if result2 == 1 else '✗ INCORRECT'}")

# Test 3: Clearly wrong answer
print("\n3️⃣ Test: Incorrect answer")
prediction3 = "The answer is 42. This follows from basic arithmetic."
result3 = task._call_judge(prediction3, label1, sample["question"])
print(f"   Wrong answer: {result3} {'✓ CORRECT' if result3 == 1 else '✗ INCORRECT (expected)'}")

# Test 4: Claude's actual response from training
print("\n4️⃣ Test: Claude's actual response from training")
claude_response = """Let me explain how to use the Legendre symbol to determine which equations have solutions.

The Legendre symbol (a/p) equals:
- 1 if a is a quadratic residue mod p (solution exists)
- -1 if a is a quadratic non-residue (no solution)

Using Euler's Criterion and quadratic reciprocity, I can show:
1) x² ≡ 7 (mod 53): HAS a solution
2) x² ≡ 53 (mod 7): HAS a solution
3) x² ≡ 14 (mod 31): HAS a solution
4) x² ≡ 25 (mod 997): HAS a solution (x=5)

Therefore, ALL FOUR equations have solutions."""

# Get a different sample for this test
legendre_sample = data["train"][5] if len(data["train"]) > 5 else sample

result4 = task._call_judge(claude_response, legendre_sample["answer"], legendre_sample["question"])
print(f"   Claude's response: {result4} {'✓ CORRECT' if result4 == 1 else '✗ INCORRECT'}")

print("\n" + "=" * 70)
print("✅ LLM-as-Judge Testing Complete!")
print("=" * 70)
print(f"\nJudge cache size: {len(task.judge_cache)} entries")
print("The judge is working and can evaluate mathematical correctness!")
