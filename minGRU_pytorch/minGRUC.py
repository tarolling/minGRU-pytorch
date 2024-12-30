import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm

from minGRU_pytorch.minGRU import minGRU


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# classes


def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner), nn.GELU(), nn.Linear(dim_inner, dim)
    )


# conv


class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value=0.0)
        x = self.net(x)
        return x.transpose(1, 2)  # b d n -> b n d


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
        labels=None,
        return_loss=False,
    ):
        x = self.token_emb(texts)

        for conv, norm, mingru, ff_norm, ff in self.layers:
            if exists(conv):
                x = conv(x) + x

            min_gru_out, _ = mingru(norm(x), return_next_prev_hidden=True)
            x = min_gru_out + x
            x = ff(ff_norm(x)) + x

        x = self.norm(x)

        if self.pooling == "mean":
            x = x.mean(dim=1)  # Mean pooling
        elif self.pooling == "max":
            x = x.max(dim=1)[0]  # Max pooling
        else:  # Default to using last hidden state
            x = x[:, -1]

        # Get logits from classification head
        logits = self.classifier(x)

        if not return_loss or labels is None:
            return logits

        # Calculate classification loss
        loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
