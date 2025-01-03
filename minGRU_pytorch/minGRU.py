# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module, Sequential, ReLU


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# appendix B
# https://github.com/glassroom/heinsen_sequence


def heinsen_associative_scan_log(log_coeffs, log_values, mask=None):
    if exists(mask):
        # Expand mask to match coefficient dimensions
        mask = mask.unsqueeze(-1).float()
        # Apply mask to coefficients (use -inf for masked positions)
        log_coeffs = log_coeffs.masked_fill(~mask.bool(), -float('inf'))
        
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    
    if exists(mask):
        # Zero out masked positions in final output
        return log_h.exp() * mask
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

    def forward(self, x, mask=None):
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
        
        if exists(mask):
            result = result * mask
        return result


# appendix B.3


def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive


class minGRU(Module):
    def __init__(self, dim, expansion_factor=1.0, substitute_ff=False, proj_out=None):
        super().__init__()
        self.substitute_ff = substitute_ff
        
        dim_inner = int(dim * expansion_factor)
        proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias=False)
        self.to_out = Linear(dim_inner, dim, bias=False) if proj_out else Identity()
        self.activation_net = ActivationNetwork(dim_inner)

    def forward(self, x, mask=None, prev_hidden=None, return_next_prev_hidden=False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)
        
        if exists(mask):
            # Expand mask to match hidden dimensions
            expanded_mask = mask.unsqueeze(-1).float()
            # Apply mask to hidden states
            hidden = hidden * expanded_mask
            gate = gate * expanded_mask

        if seq_len == 1:
            # handle sequential
            hidden = g(hidden)
            if self.substitute_ff:
                gate = self.activation_net(gate, mask)
            else:
                gate = gate.sigmoid()
                if exists(mask):
                    gate = gate * expanded_mask

            if exists(prev_hidden):
                out = torch.lerp(prev_hidden, hidden, gate)
            else:
                out = hidden * gate
        else:
            # parallel
            log_coeffs = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h
            
            if exists(prev_hidden):
                # If we have a mask, we need to handle the previous hidden state carefully
                if exists(mask):
                    # Only use prev_hidden where the first token is masked
                    first_token_mask = mask[:, 0:1]
                    prev_hidden = prev_hidden * first_token_mask.unsqueeze(-1)
                
                log_values = torch.cat((prev_hidden.log(), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))
                
                if exists(mask):
                    # Extend mask for prev_hidden
                    mask = torch.cat((torch.ones_like(mask[:, 0:1]), mask), dim=1)

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        # Get the last non-masked position for next_prev_hidden
        if exists(mask):
            # Find the last non-masked position for each sequence
            last_mask = mask.float()
            last_indices = last_mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(out.size(0), device=out.device)
            next_prev_hidden = out[batch_indices, last_indices].unsqueeze(1)
        else:
            next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden
