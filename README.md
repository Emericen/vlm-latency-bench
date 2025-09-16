# VLM Inference Latency (Tiny) Benchmark

A small test for vision language model (VLM) inference speed on text+image to text generation.

Here I compare the response speed of Qwen 2.5 VL (except 72B) vs Claude 4 Sonnet. I ran on randomized conversation of 60 turns, where each turn contains one 720p image and one short text question.

**Setup**: 2xH100 80GB SXM5 GPU for local models, Claude API for remote models.

## Experiment Commands

```bash
# Qwen2.5-VL-3B-Instruct
# On GPU node, run this:
make run MODEL=Qwen/Qwen2.5-VL-3B-Instruct TENSOR_PARALLEL=2 PORT=443 && make logs
# Locally run this:
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --data_seed 1337 --output_file results/qwen2.5_vl_3b_instruct_results_run_1.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --data_seed 66 --output_file results/qwen2.5_vl_3b_instruct_results_run_2.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-3B-Instruct --data_seed 88 --output_file results/qwen2.5_vl_3b_instruct_results_run_3.csv

# Qwen2.5-VL-3B-Instruct-AWQ
# On GPU node, run this:
make run MODEL=Qwen/Qwen2.5-VL-3B-Instruct-AWQ QUANTIZATION=awq DTYPE=half TENSOR_PARALLEL=2 PORT=443 && make logs
# Locally run this:
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-3B-Instruct-AWQ --data_seed 1337 --output_file results/qwen2.5_vl_3b_instruct_awq_results_run_1.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-3B-Instruct-AWQ --data_seed 66 --output_file results/qwen2.5_vl_3b_instruct_awq_results_run_2.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-3B-Instruct-AWQ --data_seed 88 --output_file results/qwen2.5_vl_3b_instruct_awq_results_run_3.csv

# Qwen2.5-VL-7B-Instruct
# On GPU node, run this:
make run MODEL=Qwen/Qwen2.5-VL-7B-Instruct TENSOR_PARALLEL=2 PORT=443 && make logs
# Locally run this:
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-7B-Instruct --data_seed 1337 --output_file results/qwen2.5_vl_7b_instruct_results_run_1.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-7B-Instruct --data_seed 66 --output_file results/qwen2.5_vl_7b_instruct_results_run_2.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-7B-Instruct --data_seed 88 --output_file results/qwen2.5_vl_7b_instruct_results_run_3.csv

# Qwen2.5-VL-7B-Instruct-AWQ
# On GPU node, run this:
make run MODEL=Qwen/Qwen2.5-VL-7B-Instruct-AWQ QUANTIZATION=awq DTYPE=half TENSOR_PARALLEL=2 PORT=443 && make logs
# Locally run this:
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-7B-Instruct-AWQ --data_seed 1337 --output_file results/qwen2.5_vl_7b_instruct_awq_results_run_1.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-7B-Instruct-AWQ --data_seed 66 --output_file results/qwen2.5_vl_7b_instruct_awq_results_run_2.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-7B-Instruct-AWQ --data_seed 88 --output_file results/qwen2.5_vl_7b_instruct_awq_results_run_3.csv

# Qwen2.5-VL-32B-Instruct
# On GPU node, run this:
make run MODEL=Qwen/Qwen2.5-VL-32B-Instruct TENSOR_PARALLEL=2 PORT=443 && make logs
# Locally run this:
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --data_seed 1337 --output_file results/qwen2.5_vl_32b_instruct_results_run_1.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --data_seed 66 --output_file results/qwen2.5_vl_32b_instruct_results_run_2.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-32B-Instruct --data_seed 88 --output_file results/qwen2.5_vl_32b_instruct_results_run_3.csv

# Qwen2.5-VL-32B-Instruct-AWQ
# On GPU node, run this:
make run MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ QUANTIZATION=awq DTYPE=half TENSOR_PARALLEL=2 PORT=443 && make logs
# Locally run this:
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-32B-Instruct-AWQ --data_seed 1337 --output_file results/qwen2.5_vl_32b_instruct_awq_results_run_1.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-32B-Instruct-AWQ --data_seed 66 --output_file results/qwen2.5_vl_32b_instruct_awq_results_run_2.csv
python scripts/s1_local_multi_modal.py --model_name Qwen/Qwen2.5-VL-32B-Instruct-AWQ --data_seed 88 --output_file results/qwen2.5_vl_32b_instruct_awq_results_run_3.csv

# Claude 4 Sonnet
# Locally run this:
python scripts/s2_remote_multi_modal.py --model_name claude-sonnet-4-20250514 --data_seed 1337 --output_file results/claude_sonnet_4_results_run_1.csv
python scripts/s2_remote_multi_modal.py --model_name claude-sonnet-4-20250514 --data_seed 66 --output_file results/claude_sonnet_4_results_run_2.csv
python scripts/s2_remote_multi_modal.py --model_name claude-sonnet-4-20250514 --data_seed 88 --output_file results/claude_sonnet_4_results_run_3.csv
```