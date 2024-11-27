import torch
from transformers import Trainer
import torch.nn.functional as F
import torch.nn as nn


class CustomTrainer(Trainer):
    def calc_expert_weight_norm(self, model, expert_weights):
        expert_count = len(expert_weights)
        expert_avg_norm = [0 for _ in range(expert_count)]
        for name, param in model.named_parameters():
            for expert_id in range(expert_count):
                if (f"experts.{expert_id}" in name) and (expert_weights[expert_id] > 0):
                    expert_avg_norm[expert_id] += expert_weights[expert_id] * torch.norm(param, p=2)

        for expert_id in range(expert_count):
            expert_avg_norm[expert_id] /= 32
        total_norm = sum(expert_avg_norm)
        return total_norm
    
    def compute_loss(self, model, inputs, return_outputs=False):
        exog_var = inputs["exog_var"].float()
        inputs = {k:v for k,v in inputs.items() if k != "exog_var"}

        if self.args.gate_loss_weight > 0:
            batch_size = exog_var.shape[0]
            layer_gate_weights = []
            def my_hook(module, input, output):
                routing_weights = F.softmax(output, dim=1, dtype=torch.float)
                routing_weights = routing_weights.reshape(batch_size, -1, 3)
                avg_weights = routing_weights.mean(dim=1)
                layer_gate_weights.append(avg_weights)

            hooks = []
            for i in range(32):
                h = model.model.model.layers[i].mlp.gate.register_forward_hook(my_hook)
                hooks.append(h)

        if return_outputs:
            (loss, outputs) = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False)

        if sum(self.args.expert_weights) > 0:
            reg_loss =  self.calc_expert_weight_norm(model, self.args.expert_weights)
            loss += reg_loss

        if self.args.gate_loss_weight > 0:
            gate_loss_fn = nn.CrossEntropyLoss()
            mean_gate_logits = torch.stack(layer_gate_weights).mean(0)
            gate_loss = gate_loss_fn(mean_gate_logits, exog_var)
            loss +=  self.args.gate_loss_weight * gate_loss

            for h in hooks:
                h.remove()

            layer_gate_weights.clear()
            hooks.clear()
            del mean_gate_logits

        torch.cuda.empty_cache()
        return (loss, outputs) if return_outputs else loss


