import torch
from torch import nn
from torch_geometric.nn import MessagePassing, InstanceNorm, radius_graph

from .utils import (
    L2star, L2ctr, L2ext, L2per, L2sym, L2mix,
    L2star_weighted, L2ctr_weighted, L2sym_weighted, L2per_weighted,
    L2ext_weighted, L2mix_weighted,
)


class MPNN_layer(MessagePassing):
    def __init__(self, ninp, nhid):
        super(MPNN_layer, self).__init__()
        self.ninp = ninp
        self.nhid = nhid

        self.message_net_1 = nn.Sequential(nn.Linear(2 * ninp, nhid),
                                           nn.ReLU()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                           nn.ReLU()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(ninp + nhid, nhid),
                                          nn.ReLU()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(nhid, nhid),
                                          nn.ReLU()
                                          )
        self.norm = InstanceNorm(nhid)

    def forward(self, x, edge_index, batch):
        x = self.propagate(edge_index, x=x)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j):
        message = self.message_net_1(torch.cat((x_i, x_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return update


class MPMC_net(nn.Module):
    def __init__(self, dim, nhid, nlayers, nsamples, nbatch, radius, loss_fn, weights):
        super(MPMC_net, self).__init__()
        self.enc = nn.Linear(dim,nhid)
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(MPNN_layer(nhid,nhid))
        self.dec = nn.Linear(nhid,dim)
        self.nlayers = nlayers
        self.mse = torch.nn.MSELoss()
        self.nbatch = nbatch
        self.nsamples = nsamples
        self.dim = dim

        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## random input points for transformation:
        self.x = torch.rand(nsamples * nbatch, dim).to(self.torch_device)

        self.weights = weights

        batch = torch.arange(nbatch).unsqueeze(-1).to(self.torch_device)
        batch = batch.repeat(1, nsamples).flatten()
        self.batch = batch
        self.edge_index = radius_graph(self.x, r=radius, loop=True, batch=batch).to(self.torch_device)

        all_losses = {'L2star', 'L2ctr', 'L2ext', 'L2per', 'L2sym', 'L2mix', 'L2star_weighted',
                      'L2ctr_weighted', 'L2ext_weighted', 'L2per_weighted', 'L2sym_weighted', 'L2mix_weighted'}
        if loss_fn in all_losses:
            self.loss_fn = globals()[loss_fn]
        else:
            raise ValueError(f"Loss function DNE: {loss_fn}")

    def forward(self):
        X = self.x
        edge_index = self.edge_index

        X = self.enc(X)
        for i in range(self.nlayers):
            X = self.convs[i](X,edge_index,self.batch)
        X = torch.sigmoid(self.dec(X))  ## clamping with sigmoid needed so that warnock's formula is well-defined
        X = X.view(self.nbatch, self.nsamples, self.dim)
        if self.weights is None:
            loss = torch.mean(self.loss_fn(X))
        else:
            loss = torch.mean(self.loss_fn(X, self.weights))
        return loss, X
