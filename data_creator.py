import os
import copy
import logging
import json
from typing import Dict, Sequence
from dataclasses import dataclass

import torch.nn.functional as F
import transformers
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from configs import DATA_DIR


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def check_tokens(tokenizer: transformers.PreTrainedTokenizer,
                 model:transformers.PreTrainedModel):
    new_tokens_list = {}
    if tokenizer.pad_token is None:
        new_tokens_list["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        new_tokens_list["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        new_tokens_list["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        new_tokens_list["unk_token"] = DEFAULT_UNK_TOKEN

    if new_tokens_list:
        num_new_tokens = tokenizer.add_special_tokens(new_tokens_list)
        model.resize_token_embeddings(len(tokenizer))
    else:
        num_new_tokens = 0

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        print(f"{num_new_tokens} tokens are added")



def load_safety_dataset(set_type, safe_flag=True, count=None):
    logging.info("loading safety dataset...")
    dataset = load_dataset("PKU-Alignment/BeaverTails", split=f"30k_{set_type}")
    if count is None:
        count = len(dataset)
    i, sources, targets, questions = 0, [], [], []
    for example in dataset:
        if safe_flag and not example["is_safe"]:
            continue
        instance = {"output": example["response"],
                    "instruction": example["prompt"]}
        x = PROMPT_DICT["prompt_no_input"].format_map(instance)
        sources.append(x)
        targets.append(instance["output"])
        questions.append(example["prompt"])
        i += 1
        if i >= count:
            break
    return sources, targets, questions


def load_hfl_dataset(set_type="train", count=None):
    logging.info("loading helpfulness dataset...")

    if set_type == "train":
        data_path = os.path.join(DATA_DIR, "alpaca_small.json")
        with open(data_path) as f:
            dataset = json.load(f)
    else:
        dataset = load_dataset("tatsu-lab/alpaca_eval", trust_remote_code=True)
        dataset = dataset["eval"]

    prompt_input, prompt_no_input = (PROMPT_DICT["prompt_input"],
                                      PROMPT_DICT["prompt_no_input"])    
    count = count if count is not None else len(dataset)

    i, sources, targets, questions = 0, [], [], []
    for example in dataset:
        if example.get("input", "") != "":
            sources.append(prompt_input.format_map(example))
        else:
            sources.append(prompt_no_input.format_map(example))
        targets.append(example["output"])
        questions.append(example["instruction"])
        i += 1
        if i >= count:
            break
    return sources, targets, questions


def load_truth_dataset(set_type="train", count=None):
    logging.info("loading truthful dataset")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    dataset = dataset["validation"]  # has only validation key

    dataset_dict = dataset.train_test_split(test_size=0.5, shuffle=False)
    train_ds, test_ds = dataset_dict["train"], dataset_dict["test"]

    dataset = train_ds if set_type == "train" else test_ds
    prompt = PROMPT_DICT["prompt_no_input"]

    i, sources, targets, questions = 0, [], [], []
    for example in dataset:
        if set_type == "test":
            sources.append(prompt.format(instruction=example["question"]))
            targets.append(example["best_answer"])
            questions.append(example["question"])
            i += 1
        else:
            for answer in example["correct_answers"]:
                sources.append(prompt.format(instruction=example["question"]))
                targets.append(answer)
                questions.append(example["question"])
                i += 1
        if count is not None and i >= count:
            break
    return sources, targets, questions


def tokenize_inputs(tokenizer, sources, targets):
    """
    input = text + output
    label = [IGNORE_INDEX, IGNORE_INDEX ..., IGNORE_INDEX, out_token1, out_token2, ..., out_tokenN]
    """
    # tokenize the input text which is concat of source and target
    combined_text = [s + t + "###END" for (s, t) in zip(sources, targets)]
    model_inputs = tokenizer(combined_text, padding="longest", 
                             truncation=True, return_tensors="pt", 
                             max_length=tokenizer.model_max_length)
    input_ids = model_inputs["input_ids"]
    labels = copy.deepcopy(input_ids)

    # find the starting index of the target tokens and ignore the tokens before that
    source_tokens = tokenizer(sources, padding="longest", truncation=True, return_tensors="pt")
    target_indices = source_tokens["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
    for i, idx in enumerate(target_indices):
        labels[i, :idx] = IGNORE_INDEX
    labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)



class InstructDataset(Dataset):
    def __init__(self, tokenizer, dataset_category="safety", set_type="train") -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_category = dataset_category
        self.set_type = set_type

        dataset_type = []
        if dataset_category == "safety":
            sources, targets, _ = load_safety_dataset(set_type=self.set_type)
            dataset_type = torch.ones(len(sources))
        elif dataset_category == "helpfulness":
            sources, targets, _ = load_hfl_dataset(set_type=self.set_type)
            dataset_type = torch.zeros(len(sources))
        elif dataset_category == "truthfulness":
            sources, targets, _ = load_truth_dataset(set_type=self.set_type)
            dataset_type = torch.ones(len(sources)) * 2
        elif dataset_category == "mix":
            sources, targets = [], []
            for i, method in enumerate([load_hfl_dataset, 
                                        load_safety_dataset, 
                                        load_truth_dataset]):
                s, t, _ = method(set_type=self.set_type)
                sources += s
                targets += t
                dataset_type.append(torch.ones(len(sources)) * i)
            dataset_type = torch.cat(dataset_type)
        else:
            raise KeyError("dataset_type is not recognized")

        logging.info("Tokenizing the inputs and targets...")
        self.dataset_type = F.one_hot(dataset_type.long(), num_classes=3)
        data_dict = tokenize_inputs(tokenizer=tokenizer, sources=sources, targets=targets)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[index], 
                    labels=self.labels[index], 
                    exog_var=self.dataset_type[index])


@dataclass
class DataCollatorForInstructDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, exog_var = [], [], []
        for instance in instances:
            ids, lbl = instance["input_ids"], instance["labels"]
            input_ids.append(ids)
            labels.append(lbl)
            if "exog_var" in instance.keys():
                exog_var.append(instance["exog_var"])

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return_dict  = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).int(),
        )

        if len(exog_var) > 0:
            exog_var = torch.stack(exog_var)
            return_dict["exog_var"] = exog_var

        return return_dict
