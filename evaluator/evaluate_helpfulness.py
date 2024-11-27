import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from data_creator import load_hfl_dataset


cur_dir = os.path.dirname(os.path.abspath(__file__))


def run():
    max_count = 50
    sub_dir = "cross_mix_moe_reg_10000_00000_00000_gate_00000"  # raw, aligned or cross_<task_name>
    infer_data_path = output_file_path = os.path.join(
            cur_dir, "..", "results", "inference", "helpfulness", "test", sub_dir, "outputs_final.csv")

    data_df = pd.read_csv(infer_data_path, index_col=0)
    responses = data_df["outputs"].values.tolist()

    sources, targets, questions = load_hfl_dataset("test")

    # create the json dict
    model_out_dict = []
    for q, ans in zip(questions, responses):
        ans = ans.replace("### Instruction", "").replace("\n", "").strip()
        ans = ans.replace("###END", "")
        model_out_dict.append({
            "instruction": q,
            "output": ans
        })
        if len(model_out_dict) == max_count:
            break

    file_name = os.path.join(cur_dir, "helpfulness_results", f"{sub_dir}.json")
    with open(file_name, 'w') as json_file:
        json.dump(model_out_dict, json_file, indent=4)

    print("loading reference model outputs")
    ref_name = os.path.join(cur_dir, "helpfulness_results", f"text_davinci_003_outputs.json")
    with open(ref_name, 'r') as json_file:
        ref_outputs = json.load(json_file)

    ref_name = os.path.join(cur_dir, "helpfulness_results", f"reference_model.json")
    with open(ref_name, 'w') as json_file:
        json.dump(ref_outputs[:max_count], json_file, indent=4)



if __name__ == "__main__":
    run()
