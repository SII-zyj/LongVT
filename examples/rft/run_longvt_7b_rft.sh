export NCCL_BLOCKING_WAIT=0
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER="decord"
export HF_TOKEN=""
export HF_HOME=""
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export WANDB_API_KEY=""

set -x
cd path/to/lmms-engine/
source .venv/bin/activate
wandb login $WANDB_API_KEY --relogin --host=https://api.bandw.top

# Debug mode (for local testing, comment out for cluster jobs)
torchrun --nproc_per_node "8" --master_addr "127.0.0.1" --node_rank "0" --master_port "8000" --nnodes "1" \
    -m lmms_engine.launch.cli config_yaml=path/to/longvt_7b_rft.yaml

# Task mode (for submitting jobs to cluster)
torchrun --nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST --node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
    -m lmms_engine.launch.cli config_yaml=path/to/longvt_7b_rft.yaml