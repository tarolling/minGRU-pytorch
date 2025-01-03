import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm

from minGRU_pytorch.minGRU import minGRU


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def masked_pooling(hidden_states, attention_mask, pooling_type="mean"):
    if attention_mask is None:
        if pooling_type == "mean":
            return hidden_states.mean(dim=1)
        elif pooling_type == "max":
            return hidden_states.max(dim=1)[0]
        else:  # last
            return hidden_states[:, -1]
    
    # Convert attention mask to float and unsqueeze to match hidden states
    mask = attention_mask.float().unsqueeze(-1)
    
    if pooling_type == "mean":
        # Sum up vectors and divide by the total number of non-masked tokens
        sum_masked = (hidden_states * mask).sum(dim=1)
        return sum_masked / (mask.sum(dim=1).clamp(min=1e-9))
    
    elif pooling_type == "max":
        # Set masked positions to large negative value before max
        masked_hidden = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), -1e9)
        return masked_hidden.max(dim=1)[0]
    
    else:  # last
        # Find the last non-masked position for each sequence
        last_positions = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        return hidden_states[torch.arange(batch_size), last_positions]


# classes


class FeedForward(Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Linear(dim_inner, dim)
        )
    
    def forward(self, x, mask=None):
        out = self.net(x)
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


# conv


class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)  # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value=0.0)
        x = self.net(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x

# main class


class minGRUC(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        num_classes=2,
        substitute_ff=False,
        ff_mult=4,
        min_gru_expansion=1.5,
        conv_kernel_size=3,
        enable_conv=False,
        pooling="mean"
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pooling = pooling

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        (
                            CausalDepthWiseConv1d(dim, conv_kernel_size)
                            if enable_conv
                            else None
                        ),
                        RMSNorm(dim),
                        minGRU(
                            dim,
                            expansion_factor=min_gru_expansion,
                            substitute_ff=substitute_ff,
                        ),
                        RMSNorm(dim),
                        FeedForward(dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = RMSNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_classes),
        )

        self.can_cache = not enable_conv

    def forward(
        self,
        texts,
        attention_mask=None,
        labels=None,
        return_loss=False,
    ):
        x = self.token_emb(texts)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        for conv, norm, mingru, ff_norm, ff in self.layers:
            if exists(conv):
                x = conv(x) + x

            normed = norm(x)
            if attention_mask is not None:
                normed = normed * attention_mask.unsqueeze(-1)
            
            min_gru_out, _ = mingru(
                    normed,
                    mask=attention_mask,
                    return_next_prev_hidden=True
            )
            x = min_gru_out + x
            normed = ff_norm(x)
            if attention_mask is not None:
                normed = normed * attention_mask.unsqueeze(-1)
            x = ff(normed, attention_mask) + x

        x = self.norm(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        x = masked_pooling(x, attention_mask, self.pooling)

        # Get logits from classification head
        logits = self.classifier(x)

        if not return_loss or labels is None:
            return logits

        # Calculate classification loss
        loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
