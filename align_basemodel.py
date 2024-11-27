import os
import glob
import logging
from configs import hf_token, HF_CACHE, llm_domains
os.environ['HF_HOME'] = HF_CACHE

import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from transformers import set_seed

import torch
import transformers
from peft import LoraConfig, get_peft_model
from data_creator import check_tokens, InstructDataset, DataCollatorForInstructDataset
from peft import PeftModel
from models.llama_moe import LlamaConfig, MoeLlamaForCausalLM
from custom_trainer import CustomTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Llama-2-7b-hf")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="results")
    cache_dir: Optional[str] = field(default=None)
    learning_rate: float = field(default=0.0005)
    save_steps: float = field(default=5000)
    logging_steps: float = field(default=100)
    num_train_epochs: float = field(default=3)
    model_max_length: int = field(
        default=300,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def run(model_args, data_args, training_args, exp_args):
    print(exp_args)
    set_seed(42)
    from huggingface_hub import login
    login(token=hf_token)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    data_args.dataset_category = exp_args.task_name
    training_args.optimizer = "AdamW"
    training_args.hub_token = hf_token

    if exp_args.moe_flag:
        training_args.gate_loss_weight = exp_args.gate_loss_weight
        expert_weights = [exp_args.helpfulness_weight,
                          exp_args.safety_weight,
                          exp_args.truthfulness_weight]
        training_args.expert_weights = expert_weights  # "helpfulness", "safety", "truthfulness"
        weight_str = "_".join([f"{w:.4f}".replace(".", "") for w in expert_weights])
        gate_loss_weight_str = f"{exp_args.gate_loss_weight:.4f}".replace(".", "")
        exp_args.task_name += "_moe" + f"_reg_{weight_str}_gate_{gate_loss_weight_str}"

    checkpoint_dir = os.path.join(training_args.output_dir, "checkpoints", exp_args.task_name)
    check_prev_model = False
    adapter_model_path = ""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        check_prev_model = False
    training_args.output_dir = checkpoint_dir

    if check_prev_model:
        last_checkpoint = max([int(c_dir.split("-")[1]) for c_dir in os.listdir(checkpoint_dir)])
        adapter_model_path = os.path.join(checkpoint_dir, f"checkpoint-{last_checkpoint}")

    logging.info("Loading model...")
    model_name = f"{llm_domains[model_args.model_name_or_path]}/{model_args.model_name_or_path}"
    if not exp_args.moe_flag:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            token = hf_token
        )
    else:
        configuration = LlamaConfig(max_position_embeddings=2048)
        model = MoeLlamaForCausalLM(configuration, num_experts=3, top_k=2)
        model.load_state_dict(torch.load("models/moe.pt", weights_only=True))
        model = model.half()
        model.to("cuda")
    logging.info("Model loaded...")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        token = hf_token
    )
    # tokenizer.add_eos_token = True
    check_tokens(tokenizer=tokenizer, model=model)

    train_dataset = InstructDataset(tokenizer=tokenizer,
                                    dataset_category=data_args.dataset_category,
                                    set_type="train")
    data_collator = DataCollatorForInstructDataset(tokenizer=tokenizer)

    if check_prev_model:
        logging.info(f"Loading last checkpoint from {adapter_model_path}")
        model = PeftModel.from_pretrained(
            model, 
            adapter_model_path, 
            device_map='auto', torch_dtype=torch.float16)
        model = model.merge_and_unload()
    else:
        logging.info("Initialize Lora weights...")
        if exp_args.moe_flag:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        else:
            target_modules = ["up_proj", "down_proj", "gate_proj"]
        config = LoraConfig(
            r=8,
            lora_alpha=4,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    model.train()
    if exp_args.moe_flag:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        training_args.remove_unused_columns = False
        trainer = CustomTrainer(model=model, 
                                tokenizer=tokenizer,
                                args=training_args,
                                data_collator=data_collator,
                                train_dataset=train_dataset)
    else:
        trainer = transformers.Trainer(model=model, 
                                       tokenizer=tokenizer,
                                       args=training_args,
                                       data_collator=data_collator,
                                       train_dataset=train_dataset)

    # perform training
    trainer.train()

    trainer.save_state()
    final_model_dir = os.path.join("results", "outputs", f"{exp_args.task_name}")
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)

    logging.info(f"Saving model into {final_model_dir}")
    model.save_pretrained(final_model_dir)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, 
                                            DataArguments,
                                            TrainingArguments))
    parser.add_argument("--task_name", type=str, default="mix", 
                        choices=["truthfulness", "safety", "helpfulness", "mix"])
    parser.add_argument("--moe_flag", type=int, default=1)
    parser.add_argument("--expert_topk", type=int, default=2)
    parser.add_argument("--gate_loss_weight", type=float, default=0.0)
    parser.add_argument("--helpfulness_weight", type=float, default=0.0)
    parser.add_argument("--safety_weight", type=float, default=0.0)
    parser.add_argument("--truthfulness_weight", type=float, default=0.0)
    all_args = parser.parse_args_into_dataclasses()
    run(*all_args)
