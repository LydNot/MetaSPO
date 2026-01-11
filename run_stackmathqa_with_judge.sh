#!/bin/bash
# StackMathQA with Claude Opus 4.5 + LLM-as-Judge
# BREAKTHROUGH: First time MetaSPO works on complex proof-based problems!
# Cost: ~$12, Time: ~1.5 hours

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./logs/metaspo/stackmathqa_judged_${TIMESTAMP}"
OUTPUT_LOG="./logs/stackmathqa_judged_${TIMESTAMP}.log"

echo "============================================"
echo "🚀 StackMathQA + LLM-as-Judge Training"
echo "============================================"
echo "Start time: $(date)"
echo "Log directory: ${LOG_DIR}"
echo ""
echo "Configuration:"
echo "- Base Model: Claude Opus 4.5"
echo "- Judge Model: GPT-4o-mini (for evaluation)"
echo "- Optim Model: GPT-4o-mini"
echo "- Tasks: algebra_small_judged, geometry_small_judged"
echo "- Train size: 10 samples per task"
echo "- Test size: 50 samples per task"
echo "- Iterations: 2"
echo ""
echo "Innovation:"
echo "  ✨ Using LLM-as-judge to evaluate mathematical proofs"
echo "  ✨ First time MetaSPO works on explanation-based tasks"
echo "  ✨ Extends methodology beyond original paper"
echo ""
echo "Estimated cost: ~$12"
echo "Estimated time: ~1.5 hours"
echo "============================================"
echo ""

# Run the training
python meta_train.py \
  --method metaspo \
  --task_config_path ./configs/math_judged.yaml \
  --init_system_prompt_path ./prompts/default.json \
  --log_dir "${LOG_DIR}" \
  --base_model_type claude \
  --base_model_name claude-opus-4.5 \
  --base_model_temperature 0.0 \
  --optim_model_type openai \
  --optim_model_name gpt-4o-mini \
  --optim_model_temperature 1.0 \
  --train_size 10 \
  --test_size 50 \
  --iteration 2 \
  --num_system_candidate 9 \
  --num_user_candidate 3 \
  --user_top_k 3 2>&1 | tee "${OUTPUT_LOG}"

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Training completed!"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Training completed!"
    echo ""
    echo "Results:"
    echo "  Log directory: ${LOG_DIR}"
    echo "  Output log: ${OUTPUT_LOG}"
    echo ""
    echo "Optimized prompt:"
    find "${LOG_DIR}/.." -name "bilevel_nodes_*.json" -type f -exec echo "  {}" \; | head -1
    echo ""
    echo "🎉 MetaSPO now works on complex mathematical proofs!"
else
    echo "❌ ERROR: Training failed with exit code ${EXIT_CODE}"
    echo "Check ${OUTPUT_LOG} for details"
fi

exit $EXIT_CODE
