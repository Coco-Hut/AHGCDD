from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add, scatter
from torch_geometric.utils import softmax
from torch.nn import Parameter
from torch import Tensor
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class MLPAdapter(nn.Module):
    
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden=256,
        nlayers=2,
        dropout=0.5,
        dropout_input=0,
        with_bn=0,
        residual_ratio=0,
        with_inout_ln=0,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.residual_ratio = residual_ratio
        self.dropout = nn.Dropout(dropout)
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout_output = nn.Dropout(0.1)
        self.with_bn = with_bn
        self.with_inout_ln = with_inout_ln

        self.layers = nn.ModuleList([])
        if nlayers == 1:
            self.layers.append(MyLinear(in_dim, out_dim))
        else:
            self.layers.append(MyLinear(in_dim, hidden))
            for i in range(nlayers - 2):
                self.layers.append(MyLinear(hidden, hidden))
            self.layers.append(MyLinear(hidden, out_dim))

        if with_bn:
            self.bns = torch.nn.ModuleList()
            for _ in range(nlayers - 1):
                self.bns.append(nn.BatchNorm1d(hidden))

        if with_inout_ln:
            self.ln_in = nn.LayerNorm(in_dim)
            self.ln_out = nn.LayerNorm(out_dim)

        self.scaler = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inp):

        if self.with_inout_ln:
            inp = self.ln_in(inp)

        x = self.dropout_input(inp)

        for ix, layer in enumerate(self.layers):
            x = layer(x)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                x = F.relu(x)
                x = self.dropout(x)

        if self.residual_ratio > 0:
            x = x * (1 - self.residual_ratio) + inp * self.residual_ratio
            if self.with_inout_ln:
                x = self.ln_out(x)

        return x

class HyperConv(MessagePassing):

    def __init__(self, in_channels, out_channels,**kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HyperConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = 1
        self.concat = True

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,hypericd_weight: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
        """

        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        alpha = None

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)
        D = 1.0 / D**(0.5)
        D[D == float("inf")] = 0

        B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0, dim_size=num_edges)
        B = 1.0 / B
        B[B == float("inf")] = 0
        
        x = D.unsqueeze(-1)*x
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,edge_weight=hypericd_weight,
                                size=(num_nodes, num_edges))

        self.flow = 'target_to_source'
        #out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,size=(num_edges, num_nodes))
        out = self.propagate(hyperedge_index,x=out, norm=D, alpha=alpha,edge_weight=None,size=(num_nodes, num_edges))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        H, F = self.heads, self.out_channels

        if edge_weight is not None:
            x_j=edge_weight.view(-1, 1) * x_j
        
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)