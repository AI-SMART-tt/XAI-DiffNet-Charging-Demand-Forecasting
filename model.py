import torch
import torch.nn as nn
import numpy as np


class DiffusionGraphConv(nn.Module):
    """Diffusion Graph Convolution Layer."""

    def __init__(self, rnn_units, max_diffusion_step, num_nodes, input_dim, output_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_diffusion_step = max_diffusion_step
        self.diffusion_weights = nn.Parameter(torch.FloatTensor(max_diffusion_step * 2, input_dim, output_dim))
        self.diffusion_bias = nn.Parameter(torch.FloatTensor(output_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.diffusion_weights)
        nn.init.zeros_(self.diffusion_bias)

    def forward(self, x, supports):
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.max_diffusion_step, -1)
        x_out = []
        for support in supports:
            x_support = torch.einsum('bnki,nm->bmki', x_expanded, support)
            x_out.append(x_support)
        x_out = torch.cat(x_out, dim=2)
        x_weighted = torch.einsum('bnki,kio->bno', x_out, self.diffusion_weights)
        x_weighted += self.diffusion_bias
        return x_weighted


class DCRNNCell(nn.Module):
    """Dual-Channel Diffusion Convolutional Recurrent Neural Network Cell."""

    def __init__(self, input_dim, rnn_units, max_diffusion_step, num_nodes):
        super().__init__()
        combined_dim = input_dim + rnn_units
        self.dgc_ru = DiffusionGraphConv(rnn_units, max_diffusion_step, num_nodes, combined_dim, rnn_units * 2)
        self.dgc_c = DiffusionGraphConv(rnn_units, max_diffusion_step, num_nodes, combined_dim, rnn_units)
        self.dgc2_ru = DiffusionGraphConv(rnn_units, max_diffusion_step, num_nodes, combined_dim, rnn_units * 2)
        self.dgc2_c = DiffusionGraphConv(rnn_units, max_diffusion_step, num_nodes, combined_dim, rnn_units)

    def forward(self, x, h, supports1, supports2):
        xh = torch.cat([x, h], dim=-1)
        ru1 = torch.sigmoid(self.dgc_ru(xh, supports1))
        r1, u1 = torch.chunk(ru1, 2, dim=-1)
        c1 = torch.tanh(self.dgc_c(torch.cat([x, r1 * h], dim=-1), supports1))
        ru2 = torch.sigmoid(self.dgc2_ru(xh, supports2))
        r2, u2 = torch.chunk(ru2, 2, dim=-1)
        c2 = torch.tanh(self.dgc2_c(torch.cat([x, r2 * h], dim=-1), supports2))
        update_gate = (u1 + u2) / 2
        candidate = (c1 + c2) / 2
        new_h = (1.0 - update_gate) * h + update_gate * candidate
        return new_h


class AdvancedSpatioTemporalGNN(nn.Module):
    """The main model architecture (XAI-DiffNet)."""

    def __init__(self, num_nodes, seq_len, pred_len, rnn_units, num_rnn_layers, max_diffusion_step, input_dim=1,
                 output_dim=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.rnn_units = rnn_units
        self.num_rnn_layers = num_rnn_layers
        self.max_diffusion_step = max_diffusion_step
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_fc = nn.Linear(self.input_dim, self.rnn_units)
        self.output_fc = nn.Linear(self.rnn_units, self.output_dim)

        self.encoder_cells = nn.ModuleList(
            [DCRNNCell(self.rnn_units, self.rnn_units, self.max_diffusion_step, self.num_nodes) for _ in
             range(self.num_rnn_layers)])
        self.decoder_cells = nn.ModuleList(
            [DCRNNCell(self.rnn_units, self.rnn_units, self.max_diffusion_step, self.num_nodes) for _ in
             range(self.num_rnn_layers)])

    @staticmethod
    def _get_supports_from_tensor(adj_tensor: torch.Tensor):
        device = adj_tensor.device
        adj_hat = adj_tensor + torch.eye(adj_tensor.size(0), device=device)
        d_out = torch.sum(adj_hat, dim=1)
        d_in = torch.sum(adj_hat, dim=0)
        d_inv_out = torch.pow(d_out, -1).flatten()
        d_inv_out[torch.isinf(d_inv_out)] = 0.
        d_mat_inv_out = torch.diag(d_inv_out)
        support_fwd = torch.matmul(d_mat_inv_out, adj_hat)
        d_inv_in = torch.pow(d_in, -1).flatten()
        d_inv_in[torch.isinf(d_inv_in)] = 0.
        d_mat_inv_in = torch.diag(d_inv_in)
        support_bwd = torch.matmul(d_mat_inv_in, adj_hat.T)
        return [support_fwd, support_bwd]

    def forward(self, x, adj, spatial_mask_1=None, spatial_mask_2=None, temporal_mask=None):
        batch_size = x.size(0)
        device = x.device
        if temporal_mask is not None:
            x = x * (torch.tanh(temporal_mask) + 1).unsqueeze(0).unsqueeze(-1)

        if isinstance(adj, np.ndarray):
            adj_torch = torch.from_numpy(adj).float().to(device)
        else:
            adj_torch = adj

        adj_m1 = adj_torch * (torch.tanh(
            (spatial_mask_1 + spatial_mask_1.T) / 2) + 1) if spatial_mask_1 is not None else adj_torch
        adj_m2 = adj_torch * (torch.tanh(
            (spatial_mask_2 + spatial_mask_2.T) / 2) + 1) if spatial_mask_2 is not None else adj_torch

        supports1 = self._get_supports_from_tensor(adj_m1)
        supports2 = self._get_supports_from_tensor(adj_m2)

        h_encoder = [torch.zeros(batch_size, self.num_nodes, self.rnn_units, device=device) for _ in
                     range(self.num_rnn_layers)]
        for t in range(self.seq_len):
            x_t_emb = self.input_fc(x[:, t, ...])
            h_in_layer = x_t_emb
            for i in range(self.num_rnn_layers):
                h_encoder[i] = self.encoder_cells[i](h_in_layer, h_encoder[i], supports1, supports2)
                h_in_layer = h_encoder[i]

        outputs = []
        h_decoder = h_encoder
        go_symbol = torch.zeros(batch_size, self.num_nodes, self.input_dim, device=device)
        for t in range(self.pred_len):
            x_t_dec = self.input_fc(go_symbol)
            h_in_layer = x_t_dec
            for i in range(self.num_rnn_layers):
                h_decoder[i] = self.decoder_cells[i](h_in_layer, h_decoder[i], supports1, supports2)
                h_in_layer = h_decoder[i]
            output_t = self.output_fc(h_decoder[-1])
            outputs.append(output_t)
            go_symbol = output_t

        outputs = torch.stack(outputs, dim=1)
        return outputs.squeeze(-1)