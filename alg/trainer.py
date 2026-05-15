import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ACGPFConfig


class ACGPFTrainer(nn.Module):
    def __init__(self, model, thinking_token_ids):
        super().__init__()
        self.model = model
        self.register_buffer("thinking_token_ids", thinking_token_ids, persistent=False)
        self.config = ACGPFConfig()

    def compute_mi_score(self, hidden_states, input_ids, gold_ids):
        device = hidden_states.device
        gold_embeddings = self.model.lm_head.weight[gold_ids.squeeze()]
        if gold_embeddings.dim() == 1:
            gold_embeddings = gold_embeddings.unsqueeze(0)

        probe_logits = torch.matmul(hidden_states, gold_embeddings.T)
        proxy_prob = probe_logits.mean(dim=-1)

        delta_info = torch.zeros_like(proxy_prob)
        delta_info[:, 1:] = proxy_prob[:, 1:] - proxy_prob[:, :-1]

        thinking_mask = torch.isin(input_ids, self.thinking_token_ids.to(device))
        final_mask = torch.where(thinking_mask, 1.0, 0.1)
        mi_scores = (F.relu(delta_info) * final_mask).sum(dim=-1)
        return mi_scores

    def compute_consensus_score(self, hidden_states, attention_mask):
        last_token_indices = attention_mask.sum(dim=1) - 1
        K = hidden_states.size(0)
        device = hidden_states.device
        path_embeds = hidden_states[torch.arange(K, device=device), last_token_indices]

        if hasattr(self.model, "get_base_model"):
            base_model = self.model.get_base_model()
        else:
            base_model = self.model

        last_layer = base_model.model.layers[-1]
        q_graph = last_layer.self_attn.q_proj(path_embeds)
        k_graph = last_layer.self_attn.k_proj(path_embeds)

        config = base_model.config
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        repeat_factor = num_heads // num_kv_heads
        k_graph = k_graph.repeat_interleave(repeat_factor, dim=-1)

        scale = (q_graph.size(-1)) ** -0.5
        attn_logits = torch.matmul(q_graph, k_graph.transpose(-1, -2)) * scale
        attn_weights = F.softmax(attn_logits, dim=-1)

        mask_diag = torch.eye(K, device=attn_weights.device).bool()
        attn_weights.masked_fill_(mask_diag, 0)
        consensus_scores = attn_weights.sum(dim=0)
        return consensus_scores

    def forward(
        self,
        input_ids,
        attention_mask,
        gold_input_ids,
        assistant_start_positions=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden = outputs.hidden_states[-1]
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous().float()

        if assistant_start_positions is not None:
            K, L = shift_mask.shape
            positions = torch.arange(L, device=shift_mask.device).unsqueeze(0)
            adjusted_positions = (assistant_start_positions - 1).clamp(min=0)
            label_mask = (positions >= adjusted_positions.unsqueeze(1)).float()
            shift_mask = shift_mask * label_mask

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())
        per_teacher_loss = (token_losses * shift_mask).sum(dim=1) / shift_mask.sum(
            dim=1
        ).clamp(min=1)

        mi_scores = self.compute_mi_score(last_hidden, input_ids, gold_input_ids)
        cons_scores = self.compute_consensus_score(last_hidden, attention_mask)

        def normalize(t):
            if t.numel() <= 1:
                return torch.ones_like(t)
            return (t - t.mean()) / (t.std() + 1e-4)

        norm_mi = normalize(mi_scores.detach())
        norm_cons = normalize(cons_scores.detach())
        norm_ppl = normalize(per_teacher_loss.detach())

        final_scores = (
            self.config.beta_1 * norm_mi
            + self.config.beta_2 * norm_cons
            - self.config.beta_3 * norm_ppl
        )
        alphas = F.softmax(final_scores / self.config.temperature, dim=0)

        loss_sft = (alphas.detach() * per_teacher_loss).sum()

        loss_mcc = torch.tensor(0.0, device=input_ids.device, dtype=loss_sft.dtype)
        if self.config.lambda_mcc > 0:
            K = input_ids.size(0)
            for i in range(K):
                for j in range(i + 1, K):
                    w_ij = (alphas[i] * alphas[j]).detach()
                    if w_ij < 1e-4:
                        continue
                    mask_ij = attention_mask[i] * attention_mask[j]
                    log_p_i = F.log_softmax(logits[i], dim=-1)
                    log_p_j = F.log_softmax(logits[j], dim=-1)
                    kl_fwd = F.kl_div(
                        log_p_i, log_p_j, reduction="none", log_target=True
                    ).sum(dim=-1)
                    kl_bwd = F.kl_div(
                        log_p_j, log_p_i, reduction="none", log_target=True
                    ).sum(dim=-1)
                    sym_kl = (kl_fwd + kl_bwd) * mask_ij
                    valid_tokens = mask_ij.sum().clamp(min=1)
                    loss_mcc = loss_mcc + w_ij * (sym_kl.sum() / valid_tokens)

        total_loss = loss_sft + self.config.lambda_mcc * loss_mcc
        return total_loss
