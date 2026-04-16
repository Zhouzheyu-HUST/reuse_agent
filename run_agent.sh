# 参数
provider="华科何强组"
eval="true"
include_text="true"
eval_type="single"
single_eval_w_crop=0.75
single_eval_h_crop=0.75
multi_eval_w_stage1_crop=0.75
multi_eval_h_stage1_crop=0.75
multi_eval_w_stage2_crop=0.75
multi_eval_h_stage2_crop=0.75
agent_llm="Qwen3-VL-8B-Instruct"
cache_llm="Qwen3-VL-8B-Instruct"
enable_cache="true"

# 运行
python -X utf8 run.py \
    --provider "$provider" \
    --eval "$eval" \
    --include_text "$include_text" \
    --eval_type "$eval_type" \
    --single_eval_w_crop "$single_eval_w_crop" \
    --single_eval_h_crop "$single_eval_h_crop" \
    --multi_eval_w_stage1_crop "$multi_eval_w_stage1_crop" \
    --multi_eval_h_stage1_crop "$multi_eval_h_stage1_crop" \
    --multi_eval_w_stage2_crop "$multi_eval_w_stage2_crop" \
    --multi_eval_h_stage2_crop "$multi_eval_h_stage2_crop" \
    --agent_llm "$agent_llm" \
    --cache_llm "$cache_llm" \
    --enable_cache "$enable_cache"
