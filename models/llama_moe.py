import os
import sys
os.environ['HF_HOME'] = "~/scratch/hf-cache"

sys.path.insert(0, "..")
from configs import hf_token
import torch
import transformers
from safetensors import safe_open
from torch import nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaMLP, LlamaModel, LlamaDecoderLayer, LlamaForCausalLM,
      LlamaConfig, LlamaPreTrainedModel, LlamaRMSNorm, LlamaRotaryEmbedding) 




class SparseLlamaBlock(nn.Module):
    def __init__(self, config, num_experts, top_k, router_jitter_noise=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([LlamaMLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = router_jitter_noise

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        # reshape to router_logits: (batch * sequence_length, n_experts)
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states



class MoeLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, num_experts, top_k):
        super().__init__(config, layer_idx)
        self.mlp = SparseLlamaBlock(config, num_experts, top_k)



class MoeLlamaModel(LlamaModel):
    def __init__(self, config, num_experts, top_k):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MoeLlamaDecoderLayer(config, layer_idx, num_experts, top_k) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class MoeLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config, num_experts, top_k=2):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = MoeLlamaModel(config, num_experts, top_k)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


def extract_lora(in_tensors, expert_id=0):
    new_tensors = {}
    stack = []
    for k, v in in_tensors.items():
        if not "lora" in k:
            continue
        if len(stack) == 0:
            stack.append(v)
        else:
            A = stack.pop()
            weight = torch.mm(v, A)
            weight_name = k.replace(".lora_B", "")
            weight_name = weight_name.replace("base_model.model.", "")
            weight_name = weight_name.replace("mlp", f"mlp.experts.{expert_id}")
            new_tensors[weight_name] = weight
    return new_tensors



def load_pretrained_weights(model_weights, expert_names, outputs_dir, num_experts=3):
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    temp_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        token = hf_token
    )

    # update base parameters
    for name, param in temp_model.named_parameters():
        param.requires_grad = False
        if name in model_weights.keys():
            model_weights[name] = param
            print(f"{name} is updated")

        if "mlp" in name:
            for i in range(num_experts):
                expert_name = name.replace("mlp", f"mlp.experts.{i}")
                assert(expert_name in model_weights.keys())
                model_weights[expert_name] = param
                # print(f"{expert_name} is updated")

    for i, expert_n in enumerate(expert_names):
        data_path = f"{outputs_dir}/{expert_n}/adapter_model.safetensors"
        tensors = {}
        with safe_open(data_path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        new_tensors = extract_lora(tensors, i)

        # update expert parameters
        for name, param in new_tensors.items():
            if name in model_weights.keys():
                model_weights[name] += param
                print(f"{name} is updated")
        

if __name__ == "__main__":
    configuration = LlamaConfig(max_position_embeddings=2048)
    model = MoeLlamaForCausalLM(configuration, num_experts=3)
    model = model.half() # convert model to float16
    print("model created")
    print(model)

    model_weights = model.state_dict()
    exp_names = ["helpfulness", "safety", "truthfulness"]
    out_dir = "../results/outputs"
    load_pretrained_weights(model_weights, exp_names, out_dir, num_experts=3)
    model.load_state_dict(model_weights)
    model.half()
    torch.save(model.state_dict(), "moe.pt")
