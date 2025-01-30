from minGRU_pytorch.SimpleRNN import SimpleRNN
from torch import nn
from imdb.utils import masked_pooling
from torch.nn import Module, ModuleList
import torch.nn.functional as F


class CustomRNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_tokens=512,
        pooling="mean",
        sub_nn=False,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, input_size)
        self.pooling = pooling

        self.layers = ModuleList([])

        for _ in range(1):
            self.layers.append(
                ModuleList(
                    [
                        SimpleRNN(
                            input_size, hidden_size, batch_first=True, sub_nn=sub_nn
                        )
                    ]
                )
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
        )

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

        for (rnn,) in self.layers:
            rnn_out, _ = rnn(x, mask=attention_mask)
            x = rnn_out + x

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        x = masked_pooling(x, attention_mask, self.pooling)

        logits = self.fc(x)

        if not return_loss or labels is None:
            return logits

        loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
