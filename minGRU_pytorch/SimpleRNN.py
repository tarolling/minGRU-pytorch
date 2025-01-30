from torch import nn
import torch


def exists(v):
    return v is not None


class SimpleRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        device=None,
        dtype=None,
        sub_nn=False,
    ):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        (
                            nn.Sequential(
                                nn.Linear(input_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, hidden_size),
                            )
                            if sub_nn
                            else nn.Linear(input_size, hidden_size)
                        ),
                        nn.Linear(hidden_size, hidden_size),
                        nn.Tanh() if nonlinearity == "tanh" else nn.ReLU(),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, h_0=None, mask=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        if exists(mask):
            # If input was batch_first, transpose mask to match sequence first
            if self.batch_first:
                mask = mask.transpose(0, 1)
            # Expand mask to match hidden dimensions
            expanded_mask = mask.unsqueeze(-1).float()

        if h_0 is None:
            h_0 = torch.zeros(
                batch_size, self.hidden_size, device=x.get_device()
            )  # temp fix, set to 1 layer ONLY

        h_t_minus_1 = h_0
        h_t = h_0
        output = []

        for t in range(seq_len):
            for layer, (ih, hh, activation) in enumerate(self.layers):
                h_t = activation(ih(x[t]) + hh(h_t_minus_1))
            if exists(mask):
                # Zero out hidden state at masked positions
                h_t = h_t * expanded_mask[t]
                # Maintain previous hidden state at masked positions
                h_t = h_t + h_t_minus_1 * (1 - expanded_mask[t])
            output.append(h_t)
            h_t_minus_1 = h_t

        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h_t
