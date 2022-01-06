import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatMLP(nn.Module):

    def __init__(self,
                 input_dims,
                 output_dim,
                 dimensions=[500, 500],
                 activation='relu',
                 dropout=0.):
        super(ConcatMLP, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.input_dim = sum(input_dims)
        self.dimensions = dimensions + [output_dim]
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.mlp = MLP(
            self.input_dim,
            self.dimensions,
            self.activation,
            self.dropout)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        if x[0].dim() == 3 and x[1].dim() == 2:
            x[1] = x[1].unsqueeze(1).reshape_as(x[0])
        if x[1].dim() == 3 and x[0].dim() == 2:
            x[0] = x[0].unsqueeze(1).reshape_as(x[1])
        z = torch.cat(x, dim=x[0].dim() - 1)
        z = self.mlp(z)
        return z


class MLP(nn.Module):

    def __init__(self,
                 input_dim,
                 dimensions,
                 activation='relu',
                 dropout=0.):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.dimensions = dimensions
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.linears = nn.ModuleList([nn.Linear(input_dim, dimensions[0])])
        for din, dout in zip(dimensions[:-1], dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if (i < len(self.linears) - 1):
                x = F.__dict__[self.activation](x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x
