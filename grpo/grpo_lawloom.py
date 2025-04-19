import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATA_FILE = os.environ.get("DATA_FILE", "./cashq2.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./grpo-deepseek")

# -----------------------------
# Load and preprocess dataset
# -----------------------------
print("🔄 Loading dataset...")
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
print(f"✅ Loaded {len(dataset)} samples")

# -----------------------------
# Define reward function
# -----------------------------
def reward_func(completions, **kwargs):
    # Simple reward: longer completions get higher reward
    return [float(len(c)) for c in completions]

# -----------------------------
# Load model and apply LoRA
# -----------------------------
print(f"📦 Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", trust_remote_code=True)

# Define LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # Adjust if needed
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Inject LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# Define GRPO Training Config
# -----------------------------
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    bf16=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    num_train_epochs=3,
    remove_unused_columns=False,
    label_names=["labels"],  # ✅ Required for PEFT models
    fsdp="full_shard",       # ✅ Valid strategy
    fsdp_config={
        "offload_params": True,
        "activation_checkpointing": True,
        "auto_wrap_policy": "transformer_based",  # ✅ Correct way to enable transformer wrapping
    }
)

# -----------------------------
# Initialize GRPO Trainer
# -----------------------------
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=reward_func,
    train_dataset=dataset,
)

# -----------------------------
# Start Training
# -----------------------------
print("🚀 Starting GRPO training with LoRA and FSDP...")
trainer.train()
print("✅ Training complete.")