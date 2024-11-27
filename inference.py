import os
import tqdm
import glob
import argparse
from configs import hf_token, HF_CACHE, llm_domains
os.environ['HF_HOME'] = HF_CACHE

import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from configs import hf_token
import torch.nn.functional as F
from models.llama_moe import LlamaConfig, MoeLlamaForCausalLM
from data_creator import (load_safety_dataset, load_truth_dataset,
                           load_hfl_dataset, check_tokens)


class EosListStoppingCriteria(StoppingCriteria):
    """
    ### Instruction: -> [835, 2799, 4080, 29901]
    ###END -> [835, 11794]
    """
    def __init__(self, eos_sequence = [835, 2799, 4080, 29901]):
        self.eos_sequence = eos_sequence
        self.count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        # return self.eos_sequence in last_ids
        match_ratio = 0
        for t in self.eos_sequence:
            if t in last_ids[0]:
                match_ratio += 1
        match_ratio /= len(self.eos_sequence)
        return match_ratio > 0


def get_checkpoint_number(save_dir):
    out_files = [f for f in glob.glob(os.path.join(save_dir, "*.csv")) if "final" not in f]
    if len(out_files) > 0:
        file_counts = [int(os.path.basename(f).replace(".csv", "").split("_")[1]) for f in out_files]
        latest_number = max(file_counts)
    else:
        latest_number = 0
    return latest_number 


def get_prompts(task_name, dataset_type="test"):
    if task_name == "safety":
        sources, targets, questions = load_safety_dataset(dataset_type, safe_flag=False)
    elif task_name == "truthfulness":
        sources, targets, questions = load_truth_dataset(dataset_type)
    elif task_name == "helpfulness":
        sources, targets, questions = load_hfl_dataset(dataset_type)
    else:
        raise KeyError(task_name)
    return sources, targets, questions


def save_outputs(save_dir, prompts, outputs, count=None):
    if count is not None:
        file_name = os.path.join(save_dir, f"outputs_{count}.csv")
    else:
        file_name = os.path.join(save_dir, f"outputs_final.csv")
    output_df = pd.DataFrame({"prompts": prompts[:len(outputs)],
                                "outputs": outputs})
    output_df.to_csv(file_name)


def run(args):
    print(args)
    if args.checkpoint_number == 9000 and args.task_name == "truthfulness":
        print("skipping...")
        return

    model_path = f"meta-llama/Llama-2-7b-hf"
    # model_path = os.path.join("results", args.task_name)
    if args.checkpoint_number > 0:
        cross_mode = 0
        if args.cross_task != "" :
            model_dir = args.cross_task
            args.aligned_flag = 1
            cross_mode = 1
        else:
            model_dir = args.task_name
        adapter_model_name = os.path.join("results", "checkpoints", 
                                          model_dir, 
                                          f"checkpoint-{args.checkpoint_number}")
    else:
        cross_mode = 0
        if args.cross_task != "":
            print("Warning! Cross task mode is active.")
            adapter_model_name = os.path.join("results", "outputs", args.cross_task)
            args.aligned_flag = 1
            cross_mode = 1
        else:    
            adapter_model_name = os.path.join("results", "outputs", args.task_name)
    print(model_path)

    if "mix_moe" in adapter_model_name:
        print("Loading MoE model")
        moe_flag = True
        configuration = LlamaConfig(max_position_embeddings=2048)
        model = MoeLlamaForCausalLM(configuration, num_experts=3)
        model.load_state_dict(torch.load("models/moe.pt", weights_only=True))
        model = model.half()
    else:
        moe_flag = False
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            token = hf_token
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=300,
        padding_side="right",
        use_fast=True,
        token = hf_token
    )
    check_tokens(tokenizer=tokenizer, model=model)

    if args.aligned_flag:
        model = PeftModel.from_pretrained(
            model, 
            adapter_model_name, 
            device_map='auto', torch_dtype=torch.float16)
        model = model.merge_and_unload()
        eos_sequence = [835, 11794]
    else:
        eos_sequence = [835, 2799, 4080, 29901]

    save_dir = os.path.join("results", "inference", args.task_name, args.dataset_type)
    if cross_mode:
        save_dir = os.path.join(save_dir, f"cross_{args.cross_task}")
    elif args.aligned_flag:
        save_dir = os.path.join(save_dir, "aligned")
    else:
        save_dir = os.path.join(save_dir, "raw")

    if args.checkpoint_number > 0:
        save_dir += f"_{args.checkpoint_number}"

    model = model.to("cuda")
    pipe_finetuned = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.float16},
        device_map='auto',
        max_new_tokens=512)

    prompts, ground_truths, questions = get_prompts(args.task_name, args.dataset_type)
    if args.num_samples < 0:
        args.num_samples = len(prompts)

    # the last infered data number
    # last_ep = get_checkpoint_number(save_dir)
    last_ep = 0
    assert(last_ep < args.num_samples)
    prompts = prompts[last_ep:args.num_samples]
    print(f"Starting at episode {last_ep} ending at episode {args.num_samples}...")

    stopping_criteria = StoppingCriteriaList([EosListStoppingCriteria(eos_sequence=eos_sequence)])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if moe_flag:
        class GateWeights:
            def __init__(self):
                self.layer_gate_weights = [0 for i in range(32)]
                self.layer_count = 0

        gate_weights = GateWeights()
        def my_hook(module, input, output):
            num_tokens = output.shape[0]
            routing_weights = F.softmax(output, dim=1, dtype=torch.float)
            avg_weights = routing_weights.sum(dim=0) / num_tokens
            gate_weights.layer_gate_weights[gate_weights.layer_count % 32] += avg_weights
            gate_weights.layer_count += + 1

        for i in range(32):
            model.model.layers[i].mlp.gate.register_forward_hook(my_hook)

    outputs = []
    save_freq, count = 500, last_ep
    for prt in tqdm.tqdm(prompts):
        output = pipe_finetuned(
            prt,
            temperature=0.6,
            add_special_tokens=True,
            stopping_criteria=stopping_criteria,
            do_sample=True
        )
        outputs.append(output[0]["generated_text"][len(prt):])
        count += 1
        if count % save_freq == 0:
            save_outputs(save_dir, prompts, outputs, count=count)

    print(f"Saving all the outputs to {save_dir}")
    save_outputs(save_dir, prompts, outputs)

    if moe_flag:
        total_token_count = gate_weights.layer_count / 32
        for i in range(32):
            gate_weights.layer_gate_weights[i] = gate_weights.layer_gate_weights[i] / total_token_count

        layer_gate_weights = torch.stack(gate_weights.layer_gate_weights).cpu().numpy()
        fig, ax = plt.subplots()
        ax.imshow(layer_gate_weights)
        fig_path = os.path.join("results", "observation", f"{args.task_name}_{args.cross_task}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        arr_path = os.path.join("results", "observation", f"{args.task_name}_{args.cross_task}.npy")
        with open(arr_path, 'wb') as f:
            np.save(f, layer_gate_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference scripts for the trained models')
    parser.add_argument("--task_name", type=str, default="helpfulness", 
                        choices=["truthfulness", "safety", "helpfulness"])
    parser.add_argument("--dataset_type", type= str, default="test", choices=["test", "train"])
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--cross_task", type=str, default="mix_moe_reg_10000_00000_00000_gate_00000")
    parser.add_argument("--checkpoint_number", type=int, default=-1)
    parser.add_argument("--aligned_flag", type=int, default=1, 
                        help="if 1 loads alligned model else loads raw model")
    arguments = parser.parse_args()
    run(arguments)
