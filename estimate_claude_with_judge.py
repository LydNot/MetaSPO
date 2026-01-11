#!/usr/bin/env python3
"""
Cost estimator for MetaSPO with Claude + LLM-as-judge
"""

def estimate_with_judge(
    num_tasks=2,
    train_size=10,
    test_size=50,
    iterations=2,
    num_user_candidate=3,
    user_top_k=3,
):
    """
    Estimate cost with Claude Opus 4.5 + GPT-4o-mini judge.
    """

    # Claude Opus 4.5 pricing
    CLAUDE_INPUT = 15.0
    CLAUDE_OUTPUT = 75.0

    # GPT-4o-mini pricing (for both optimizer and judge)
    GPT_INPUT = 0.150
    GPT_OUTPUT = 0.600

    print("=" * 60)
    print("MetaSPO Cost with LLM-as-Judge")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Tasks: {num_tasks}")
    print(f"  Train size: {train_size} per task")
    print(f"  Test size: {test_size} per task")
    print(f"  Iterations: {iterations}")

    # === CLAUDE (BASE MODEL) ===
    AVG_QUESTION_TOKENS = 150
    AVG_ANSWER_TOKENS = 400
    AVG_SYSTEM_PROMPT_TOKENS = 50
    AVG_USER_PROMPT_TOKENS = 100

    # Initial evaluation
    initial_calls = num_tasks * train_size
    claude_initial_in = initial_calls * (AVG_SYSTEM_PROMPT_TOKENS + AVG_USER_PROMPT_TOKENS + AVG_QUESTION_TOKENS)
    claude_initial_out = initial_calls * AVG_ANSWER_TOKENS

    # Per iteration
    inner_loop_calls = num_tasks * num_user_candidate * train_size
    outer_loop_calls = num_tasks * user_top_k * train_size
    iter_calls = inner_loop_calls + outer_loop_calls

    claude_iter_in = iter_calls * (AVG_SYSTEM_PROMPT_TOKENS + AVG_USER_PROMPT_TOKENS + AVG_QUESTION_TOKENS)
    claude_iter_out = iter_calls * AVG_ANSWER_TOKENS
    claude_all_iter_in = claude_iter_in * iterations
    claude_all_iter_out = claude_iter_out * iterations

    # Final test
    final_calls = num_tasks * test_size
    claude_final_in = final_calls * (AVG_SYSTEM_PROMPT_TOKENS + AVG_USER_PROMPT_TOKENS + AVG_QUESTION_TOKENS)
    claude_final_out = final_calls * AVG_ANSWER_TOKENS

    # Total Claude
    claude_total_in = claude_initial_in + claude_all_iter_in + claude_final_in
    claude_total_out = claude_initial_out + claude_all_iter_out + claude_final_out
    claude_cost = (claude_total_in / 1_000_000) * CLAUDE_INPUT + (claude_total_out / 1_000_000) * CLAUDE_OUTPUT

    # === GPT-4O-MINI (JUDGE) ===
    # Judge every Claude response during training
    AVG_JUDGE_PROMPT_TOKENS = 300  # Question + ground truth + prediction
    AVG_JUDGE_OUTPUT_TOKENS = 5    # Just "CORRECT" or "INCORRECT"

    total_judge_calls = initial_calls + (iter_calls * iterations) + final_calls
    judge_total_in = total_judge_calls * AVG_JUDGE_PROMPT_TOKENS
    judge_total_out = total_judge_calls * AVG_JUDGE_OUTPUT_TOKENS
    judge_cost = (judge_total_in / 1_000_000) * GPT_INPUT + (judge_total_out / 1_000_000) * GPT_OUTPUT

    # === GPT-4O-MINI (OPTIMIZER) ===
    AVG_OPTIM_INPUT_TOKENS = 2000
    AVG_OPTIM_OUTPUT_TOKENS = 500

    inner_optim_calls = num_tasks * iterations
    outer_optim_calls = 1 * iterations
    total_optim_calls = inner_optim_calls + outer_optim_calls

    optim_total_in = total_optim_calls * AVG_OPTIM_INPUT_TOKENS
    optim_total_out = total_optim_calls * AVG_OPTIM_OUTPUT_TOKENS
    optim_cost = (optim_total_in / 1_000_000) * GPT_INPUT + (optim_total_out / 1_000_000) * GPT_OUTPUT

    # === TOTALS ===
    total_cost = claude_cost + judge_cost + optim_cost

    print(f"\n[CLAUDE OPUS 4.5 - Base Model]")
    print(f"  Calls: {initial_calls + iter_calls * iterations + final_calls}")
    print(f"  Cost: ${claude_cost:.2f}")

    print(f"\n[GPT-4o-mini - LLM Judge]")
    print(f"  Calls: {total_judge_calls}")
    print(f"  Cost: ${judge_cost:.2f}")

    print(f"\n[GPT-4o-mini - Optimizer]")
    print(f"  Calls: {total_optim_calls}")
    print(f"  Cost: ${optim_cost:.2f}")

    print(f"\n" + "=" * 60)
    print(f"TOTAL COST ESTIMATE")
    print(f"=" * 60)
    print(f"  Claude (base): ${claude_cost:.2f}")
    print(f"  Judge (eval): ${judge_cost:.2f}")
    print(f"  Optimizer: ${optim_cost:.2f}")
    print(f"  TOTAL: ${total_cost:.2f}")
    print(f"\n  Range: ${total_cost * 0.8:.2f} - ${total_cost * 1.2:.2f}")
    print("=" * 60)

    return total_cost

if __name__ == "__main__":
    print("\n📊 MINIMAL CONFIG (Fast, Cheap)")
    estimate_with_judge(
        num_tasks=2,
        train_size=5,
        test_size=20,
        iterations=1,
    )

    print("\n\n📊 RECOMMENDED CONFIG (Balanced)")
    estimate_with_judge(
        num_tasks=2,
        train_size=10,
        test_size=50,
        iterations=2,
    )
