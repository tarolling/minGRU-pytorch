# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module, Sequential, ReLU


def exists(v):
    return v is not None


# appendix B
# https://github.com/glassroom/heinsen_sequence


def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()


class ActivationNetwork(Module):
    def __init__(self, dim):
        super().__init__()
        self.net = Sequential(
            Linear(dim, dim * 2),
            ReLU(),
            Linear(dim * 2, dim),
            ReLU(),
            Linear(dim, 1),
        )

    def forward(self, x):
        # Reshape input if necessary
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])

        # Process through neural network
        result = self.net(x)

        # Ensure output is between 0 and 1 using sigmoid
        result = torch.sigmoid(result)

        # Reshape back to original dimensions
        result = result.reshape(original_shape[:-1] + (1,)).squeeze(-1)
        return result


# appendix B.3


def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive


class minGRU(Module):
    def __init__(self, dim, expansion_factor=1.0):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias=False)
        self.to_out = (
            Linear(dim_inner, dim, bias=False)
            if expansion_factor != 1.0
            else Identity()
        )
        self.activation_net = ActivationNetwork(dim_inner)

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = self.activation_net(gate)
            out = (
                torch.lerp(prev_hidden, hidden, gate)
                if exists(prev_hidden)
                else (hidden * gate)
            )
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden
