python eval/api_request_parallel_processor.py \
  --requests_filepath results/inputs/${2}-${1}.jsonl \
  --save_filepath results/outputs/${2}-${1}.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --max_attempts 5 \
  --logging_level 20 \
  --api_key ${YOUR_API_KEY}

python eval/api_request_parallel_processor.py \
  --requests_filepath results/inputs/${1}-${2}.jsonl \
  --save_filepath results/outputs/${1}-${2}.jsonl \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --max_attempts 5 \
  --logging_level 20 \
  --api_key ${YOUR_API_KEY}

python eval/grading.py --input1 results/outputs/${1}-${2}.jsonl --input2 results/outputs/${2}-${1}.jsonl --pairwise