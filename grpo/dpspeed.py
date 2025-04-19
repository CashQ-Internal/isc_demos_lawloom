# train_grpo_llama8b.py

import os
import subprocess
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

# Print GPU info
# print("🔍 Running `nvidia-smi` to check available GPUs:")
# try:
#     subprocess.run(["nvidia-smi"], check=True)
# except Exception as e:
#     print(f"⚠️ Failed to run nvidia-smi: {e}")

# Environment variables set by ISC / Accelerate
rank = int(os.environ.get("RANK", -1))
local_rank = int(os.environ.get("LOCAL_RANK", -1))
world_size = int(os.environ.get("WORLD_SIZE", -1))
n_proc = os.environ.get("N_PROC", "N/A")
n_nodes = os.environ.get("NNODES", "N/A")
master_addr = os.environ.get("MASTER_ADDR", "N/A")
print("\n🔧 Environment Variables:")
print(f"  RANK          = {rank}")
print(f"  LOCAL_RANK    = {local_rank}")
print(f"  WORLD_SIZE    = {world_size}")
print(f"  N_PROC        = {n_proc}")
print(f"  NNODES        = {n_nodes}")
print(f"  MASTER_ADDR   = {master_addr}")
print(f"\n✅ [RANK {rank}] Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
# Set the correct GPU for this process
# torch.cuda.set_device(local_rank)



# ---------------------
# Paths and Constants
# ---------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
DATA_FILE = os.environ.get("DATA_FILE", "/root/isc-demos/deepseek/cashq2.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/root/isc-demos/deepseek/grpo-llama8b")

# ---------------------
# Load and Preprocess Dataset
# ---------------------
dataset = load_dataset("json", data_files=DATA_FILE)["train"]


def preprocess(example):
    messages = example["messages"]
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    return {
        "prompt": user_msg.strip(),
        "completion": assistant_msg.strip()
    }


dataset = dataset.map(preprocess).remove_columns(dataset.column_names)

# ---------------------
# Reward Function
# ---------------------
# def reward_func(completions, **kwargs):
#     return [float(len(c)) for c in completions]

def print_available_gpus():
    if not torch.cuda.is_available():
        print("❌ CUDA is not available.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"✅ Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

from transformers import pipeline

reward_model = pipeline("text-classification", model="nlpaueb/legal-bert-base-uncased")


def reward_func(completions, **kwargs):
    results = reward_model(completions)
    return [res["score"] for res in results]


# ---------------------
# Load Tokenizer & Model
# ---------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", trust_remote_code=True,)

# ---------------------
# Apply LoRA
# ---------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load checkpoint directory from environment
try:
    output_directory = os.environ.get("CHECKPOINT_ARTIFACT_PATH", "./grpo-llama8b")
except KeyError:
    print("❌ Must set env var CHECKPOINT_ARTIFACT_PATH so we know where to save/load checkpoints!")
    exit(1)

from transformers.trainer_utils import get_last_checkpoint

# Check if a previous checkpoint exists
last_checkpoint = get_last_checkpoint(output_directory)
if last_checkpoint is not None:
    print(f"🔁 Resuming from checkpoint: {last_checkpoint}")
else:
    print("🚀 No checkpoint found. Starting fresh training run.")

# ---------------------
# GRPO + DeepSpeed Config
# ---------------------
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=4,
    # num_generations=4,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    num_train_epochs=3,
    remove_unused_columns=False,
    label_names=["labels"],
)

# ---------------------
# Launch GRPO Trainer
# ---------------------
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=reward_func,
    train_dataset=dataset,
)
print_available_gpus()
print("🚀 Starting GRPO training with LoRA + DeepSpeed...")
# trainer.train()
trainer.train(resume_from_checkpoint=last_checkpoint)
print("✅ Training complete.")
