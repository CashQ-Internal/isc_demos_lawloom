"""
FSDP implementation of Group Relative Policy Optimization (GRPO)

This file implements the GRPO algorithm introduced by DeepSeek in their DeepSeekMath paper,
using PyTorch's Fully Sharded Data Parallel (FSDP) for distributed training.

GRPO simplifies reinforcement learning by eliminating the value model and instead
estimating advantages from grouped outputs, making it more memory-efficient.
"""

import os
import functools
import logging
import warnings
import time
import random
from typing import List, Dict, Tuple, Any, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)
from peft import LoraModel, LoraConfig, get_peft_model

from datasets import load_dataset

# Import utility functions from existing codebase
from cycling_utils import AtomicDirectory, atomic_torch_save, TimestampedTimer, InterruptableDistributedSampler
from fsdp_utils import bfSixteen_ready, bfSixteen_policy, count_trainable_parameters, AppState, get_args_parser

# Initialize timer and suppress warnings
timer = TimestampedTimer("Start")
logger = logging.getLogger("torch.distributed.fsdp._state_dict_utils")
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
ADAPTER_NAME = "GRPOLora"
SHARD_STRATEGY = ShardingStrategy.FULL_SHARD
GROUP_SIZE = 4  # Number of completions to generate per prompt
CLIP_RATIO = 0.2  # Similar to PPO clip ratio

class GRPOTrainer:
    """
    Trainer class for Group Relative Policy Optimization (GRPO) using FSDP.
    
    GRPO eliminates the value model used in PPO, instead estimating advantages
    from grouped outputs, making it more memory-efficient.
    """
    
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        reward_model: Optional[PreTrainedModel] = None,
        tokenizer: PreTrainedTokenizer = None,
        device_mesh = None,
        group_size: int = GROUP_SIZE,
        clip_ratio: float = CLIP_RATIO,
        learning_rate: float = 1e-5,
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device_mesh = device_mesh
        self.group_size = group_size
        self.clip_ratio = clip_ratio
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), 
            lr=self.learning_rate
        )
        
    def generate_completions(
        self, 
        prompts: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate multiple completions for each prompt in the batch.
        
        Args:
            prompts: Tensor of shape [batch_size, seq_len] containing input prompts
            attention_mask: Attention mask for the prompts
            
        Returns:
            Tuple of (completions, completion_masks)
        """
        batch_size = prompts.shape[0]
        all_completions = []
        all_completion_masks = []
        
        # Set policy model to eval mode for generation
        self.policy_model.eval()
        
        with torch.no_grad():
            for _ in range(self.group_size):
                # Generate completions with some randomness for exploration
                outputs = self.policy_model.generate(
                    input_ids=prompts,
                    attention_mask=attention_mask,
                    max_new_tokens=128,  # Adjust based on your needs
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Extract only the new tokens (completions)
                completions = outputs[:, prompts.shape[1]:]
                
                # Create attention mask for completions
                completion_mask = torch.ones_like(completions)
                
                all_completions.append(completions)
                all_completion_masks.append(completion_mask)
        
        # Set policy model back to train mode
        self.policy_model.train()
        
        return all_completions, all_completion_masks
    
    def compute_rewards(
        self, 
        prompts: torch.Tensor, 
        completions: List[torch.Tensor],
        prompt_mask: torch.Tensor,
        completion_masks: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute rewards for each completion.
        
        Args:
            prompts: Tensor of shape [batch_size, prompt_len] containing input prompts
            completions: List of tensors, each of shape [batch_size, completion_len]
            prompt_mask: Attention mask for prompts
            completion_masks: List of attention masks for completions
            
        Returns:
            Tensor of shape [batch_size, group_size] containing rewards
        """
        batch_size = prompts.shape[0]
        device = prompts.device
        rewards = torch.zeros(batch_size, self.group_size, device=device)
        
        # If a reward model is provided, use it
        if self.reward_model is not None:
            with torch.no_grad():
                for i, (completion, completion_mask) in enumerate(zip(completions, completion_masks)):
                    # Concatenate prompts and completions
                    full_input = torch.cat([prompts, completion], dim=1)
                    full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
                    
                    # Get reward scores
                    reward_outputs = self.reward_model(
                        input_ids=full_input,
                        attention_mask=full_mask
                    )
                    rewards[:, i] = reward_outputs.rewards.squeeze(-1)
        else:
            # Placeholder: implement your own reward function if no reward model
            # For example, you might use a heuristic or external API
            for i in range(self.group_size):
                # Example: random rewards for demonstration
                rewards[:, i] = torch.rand(batch_size, device=device)
        
        return rewards
    
    def compute_log_probs(
        self, 
        model: PreTrainedModel,
        prompts: torch.Tensor, 
        completions: List[torch.Tensor],
        prompt_mask: torch.Tensor,
        completion_masks: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute log probabilities of completions under the given model.
        
        Args:
            model: The model to compute log probs with
            prompts: Tensor of shape [batch_size, prompt_len] containing input prompts
            completions: List of tensors, each of shape [batch_size, completion_len]
            prompt_mask: Attention mask for prompts
            completion_masks: List of attention masks for completions
            
        Returns:
            Tensor of shape [batch_size, group_size] containing log probs
        """
        batch_size = prompts.shape[0]
        device = prompts.device
        log_probs = torch.zeros(batch_size, self.group_size, device=device)
        
        for i, (completion, completion_mask) in enumerate(zip(completions, completion_masks)):
            # Prepare inputs for the model
            full_input = torch.cat([prompts, completion[:, :-1]], dim=1)
            full_mask = torch.cat([prompt_mask, completion_mask[:, :-1]], dim=1)
            
            # Target is the completion shifted right
            target = completion
            
            # Get model outputs
            outputs = model(
                input_ids=full_input,
                attention_mask=full_mask,
                labels=target
            )
            
            # Compute log probs from loss
            # Note: this is a simplification; in practice, you'd compute token-level log probs
            log_probs[:, i] = -outputs.loss
        
        return log_probs
    
    def compute_advantages(
        self, 
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute advantages using group-based baseline.
        
        In GRPO, we don't use a value model. Instead, we compute advantages
        by comparing each completion's reward to the average reward in its group.
        
        Args:
            rewards: Tensor of shape [batch_size, group_size] containing rewards
            
        Returns:
            Tensor of shape [batch_size, group_size] containing advantages
        """
        # Compute mean reward per group (per prompt)
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        
        # Compute advantages as reward - baseline
        advantages = rewards - mean_rewards
        
        # Normalize advantages for stability
        if advantages.numel() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def grpo_loss(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GRPO loss.
        
        Args:
            policy_log_probs: Log probs under policy model [batch_size, group_size]
            ref_log_probs: Log probs under reference model [batch_size, group_size]
            advantages: Advantages [batch_size, group_size]
            
        Returns:
            GRPO loss
        """
        # Compute probability ratio between policy and reference model
        # exp(policy_log_prob - ref_log_prob) = policy_prob / ref_prob
        ratios = torch.exp(policy_log_probs - ref_log_probs)
        
        # Clip ratios to prevent extreme policy updates
        clipped_ratios = torch.clamp(
            ratios, 
            min=1.0 - self.clip_ratio, 
            max=1.0 + self.clip_ratio
        )
        
        # Compute surrogate objectives
        surrogate1 = ratios * advantages
        surrogate2 = clipped_ratios * advantages
        
        # Take minimum to create pessimistic (clipped) objective
        surrogate_loss = -torch.min(surrogate1, surrogate2).mean()
        
        return surrogate_loss
    
    def train_step(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single GRPO training step.
        
        Args:
            prompts: Tensor of shape [batch_size, seq_len] containing input prompts
            attention_mask: Attention mask for the prompts
            
        Returns:
            Dictionary with training metrics
        """
        # 1. Generate multiple completions per prompt
        completions, completion_masks = self.generate_completions(
            prompts, attention_mask
        )
        
        # 2. Compute rewards for each completion
        rewards = self.compute_rewards(
            prompts, completions, attention_mask, completion_masks
        )
        
        # 3. Compute log probs under policy and reference models
        with torch.no_grad():
            ref_log_probs = self.compute_log_probs(
                self.ref_model, prompts, completions, attention_mask, completion_masks
            )
        
        policy_log_probs = self.compute_log_probs(
            self.policy_model, prompts, completions, attention_mask, completion_masks
        )
        
        # 4. Compute advantages using group-based baseline (no value model)
        advantages = self.compute_advantages(rewards)
        
        # 5. Compute GRPO loss
        loss = self.grpo_loss(policy_log_probs, ref_log_probs, advantages)
        
        # 6. Perform optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return metrics
        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
        }


def preprocess(example):
    """
    Preprocess a single example from the dataset.
    """
    if "question" in example and "answer" in example:
        # For QA datasets
        prompt = f"Question: {example['question']}\nAnswer:"
        completion = example["answer"]
    elif "instruction" in example and "output" in example:
        # For instruction datasets
        prompt = example["instruction"]
        completion = example["output"]
    else:
        # Fallback
        prompt = example.get("text", "")
        completion = ""
    
    return {
        "prompt": prompt,
        "completion": completion
    }


def preprocess_function(examples):
    """
    Tokenize and prepare examples for the model.
    """
    prompts = examples["prompt"]
    completions = examples["completion"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        prompts,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    
    # Tokenize targets
    labels = tokenizer(
        completions,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    
    # Create labels tensor
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
        "labels": torch.stack([torch.tensor(x["labels"]) for x in batch])
    }


if __name__ == "__main__":
    # Parse arguments
    args = get_args_parser().parse_args()
    
    # Set up distributed training
    rank = int(os.environ["RANK"])  # Global rank
    local_device = int(os.environ["LOCAL_RANK"])  # Rank on local node
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of global ranks
    model_path = os.path.join("/data", args.dataset_id)
    torch.cuda.set_device(local_device)
    
    timer.report(f"Init process group for world size: {world_size}")
    
    # Create device mesh for FSDP
    device_mesh = init_device_mesh("cuda", (world_size,))
    assert bfSixteen_ready(), "ERROR: System not BF16 ready."
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure padding token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load policy model (the model being trained)
    if rank == 0:
        policy_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_cache=False,
            torch_dtype=torch.bfloat16
        )
        print(f"Main rank {rank} policy model params on device: {set([p.data.device for p in policy_model.parameters()])}")
    else:
        with torch.device("meta"):
            policy_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                use_cache=False,
                torch_dtype=torch.bfloat16
            )
            print(f"Non-main rank {rank} policy model params on device: {set([p.data.device for p in policy_model.parameters()])}")
    
    timer.report(f"Loaded policy model: {count_trainable_parameters(policy_model)}")
    
    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
    )
    
    policy_model = LoraModel(policy_model, lora_config, ADAPTER_NAME)
    
    timer.report(f"Applied LoRA to policy model: {count_trainable_parameters(policy_model)}")
    
    # Load reference model (frozen copy of the original model)
    # In GRPO, we need a reference model to compute probability ratios
    with torch.device("meta"):
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_cache=False,
            torch_dtype=torch.bfloat16
        )
    
    timer.report("Loaded reference model")
    
    # Wrap policy model with FSDP
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000
    )
    
    policy_model = FSDP(
        policy_model,
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=SHARD_STRATEGY,
        mixed_precision=bfSixteen_policy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(),
        param_init_fn=lambda mod: mod.to_empty(device=torch.cuda.current_device(), recurse=False),
        device_mesh=device_mesh
    )
    
    # Wrap reference model with FSDP
    ref_model = FSDP(
        ref_model,
        auto_wrap_policy=my_auto_wrap_policy,
        sharding_strategy=SHARD_STRATEGY,
        mixed_precision=bfSixteen_policy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(),
        param_init_fn=lambda mod: mod.to_empty(device=torch.cuda.current_device(), recurse=False),
        device_mesh=device_mesh
    )
    
    # Freeze reference model parameters
    for param in ref_model.parameters():
        param.requires_grad = False
    
    timer.report("FSDP wrapped models")
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        device_mesh=device_mesh,
        group_size=GROUP_SIZE,
        clip_ratio=CLIP_RATIO,
        learning_rate=1e-5,
    )
    
    # Prepare dataset
    dataset = load_dataset("json", data_files=f"{model_path}/cashq2.jsonl")
    dataset = dataset["train"].map(preprocess)
    
    # Process dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing and preprocessing dataset",
    )
    
    # Create dataloader
    train_sampler = InterruptableDistributedSampler(tokenized_dataset)
    
    batch_size = 2  # Smaller batch size due to multiple completions per prompt
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler
    )
    
    steps_per_epoch = len(dataloader)
    
    # Set up checkpointing
    try:
        output_directory = os.environ["CHECKPOINT_ARTIFACT_PATH"]
    except KeyError as error:
        print("Must set env var CHECKPOINT_ARTIFACT_PATH so we know where to save checkpoints!")
        exit(1)
    
    saver = AtomicDirectory(output_directory=output_directory, is_master=rank==0)
    
    # Load checkpoint if available
    best_loss = float("inf")
    latest_symlink_file_path = os.path.join(output_directory, saver.symlink_name)
    
    if os.path.islink(latest_symlink_file_path):
        latest_checkpoint_path = os.readlink(latest_symlink_file_path)
        
        state_dict = {"app": AppState(policy_model, trainer.optimizer)}
        dcp.load(state_dict=state_dict, checkpoint_id=latest_checkpoint_path)
        
        train_state = torch.load(os.path.join(latest_checkpoint_path, "train_state.pt"))
        dataloader.sampler.load_state_dict(train_state["sampler"])
        best_loss = train_state["best_loss"]
        
        timer.report("Loaded checkpoint")
    
    state_dict = {"app": AppState(policy_model, trainer.optimizer)}
    
    # Training loop
    num_epochs = 10
    save_every_steps = 30
    policy_model.train()
    
    for epoch in range(dataloader.sampler.epoch, num_epochs):
        dataloader.sampler.set_epoch(epoch)
        
        for batch in dataloader:
            step = dataloader.sampler.progress // dataloader.batch_size
            is_last_step = (step + 1) == steps_per_epoch
            is_save_step = ((step + 1) % save_every_steps == 0) or is_last_step
            
            # Move batch to device
            input_ids = batch["input_ids"].to(torch.cuda.current_device())
            attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
            
            # Perform GRPO training step
            metrics = trainer.train_step(input_ids, attention_mask)
            loss = metrics["loss"]
            
            # Update dataloader sampler
            dataloader.sampler.advance(len(input_ids))
            
            # Synchronize loss across all ranks
            sync_loss = torch.tensor(loss, device=torch.cuda.current_device())
            dist.all_reduce(sync_loss)
            
            timer.report(f"Step {step} Loss: {sync_loss.item():.3f} Reward: {metrics['mean_reward']:.3f}")
            
            # Save checkpoint if needed
            if is_save_step:
                force_save = False
                
                checkpoint_directory = saver.prepare_checkpoint_directory(force_save=force_save)
                checkpoint_writer = dcp.FileSystemWriter(checkpoint_directory)
                
                metadata = dcp.save(
                    state_dict=state_dict,
                    storage_writer=checkpoint_writer
                )
                
                dist.barrier()
                
                if rank == 0:
                    atomic_torch_save(
                        {
                            "sampler": dataloader.sampler.state_dict(),
                            "best_loss": best_loss
                        },
                        os.path.join(checkpoint_directory, "train_state.pt")
                    )
                
                while len(os.listdir(checkpoint_directory)) < world_size + 2:
                    print("Checkpoint not yet saved...")
                    time.sleep(1)
                    dist.barrier()
                
                saver.symlink_latest(checkpoint_directory)
                
                timer.report("Saved checkpoint")
        
        dataloader.sampler.reset_progress()
    
    timer.report("Training complete")
    
    dist.barrier()
    dist.destroy_process_group()