import torch
from torch import nn
from torch_geometric.nn import MessagePassing, InstanceNorm, radius_graph

from .utils import (
    L2star, L2ctr, L2ext, L2per, L2sym, L2mix,
    L2star_weighted, L2ctr_weighted, L2sym_weighted, L2per_weighted,
    L2ext_weighted, L2mix_weighted,
)


class MPNN_layer(MessagePassing):
    """One message-passing neural network layer used by `MPMC_net`.

    Implements `torch_geometric.nn.MessagePassing`'s `message`/`update`
    interface: each node aggregates messages from its neighbors (from the
    `edge_index` graph built in `MPMC_net`) and updates its own features.
    """

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
        """Propagate messages over the graph and instance-normalize the result.

        Args:
            x (torch.Tensor): Node features, shape `(num_nodes, ninp)`.
            edge_index (torch.Tensor): Graph connectivity, shape `(2, num_edges)`.
            batch (torch.Tensor): Batch assignment for each node, shape `(num_nodes,)`.

        Returns:
            torch.Tensor: Updated, normalized node features, shape `(num_nodes, nhid)`.
        """
        x = self.propagate(edge_index, x=x)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j):
        """Compute the message sent from neighbor `x_j` to node `x_i`.

        Called internally by `MessagePassing.propagate`.

        Args:
            x_i (torch.Tensor): Features of the target node, shape `(num_edges, ninp)`.
            x_j (torch.Tensor): Features of the source (neighbor) node, shape `(num_edges, ninp)`.

        Returns:
            torch.Tensor: Message for each edge, shape `(num_edges, nhid)`.
        """
        message = self.message_net_1(torch.cat((x_i, x_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        """Combine a node's aggregated message with its own features.

        Called internally by `MessagePassing.propagate`.

        Args:
            message (torch.Tensor): Aggregated incoming message, shape `(num_nodes, nhid)`.
            x (torch.Tensor): The node's own features, shape `(num_nodes, ninp)`.

        Returns:
            torch.Tensor: Updated node features, shape `(num_nodes, nhid)`.
        """
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return update


class MPMC_net(nn.Module):
    """Graph neural network that transforms random points into a
    low-discrepancy point set by minimizing a discrepancy-based loss.

    Encodes `nbatch` independent batches of `nsamples` random points in
    `dim` dimensions, passes them through `nlayers` `MPNN_layer`s connected
    by a radius graph, decodes back to `dim` dimensions, and squashes to
    `[0,1]^dim` via a sigmoid. Trained (elsewhere, e.g. `MPMC._train`) to
    minimize `loss_fn` evaluated on the resulting points.
    """

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
        """Transform the stored random points and compute the discrepancy loss.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(loss, X)` where `loss` is
                the scalar mean discrepancy loss (weighted by `self.weights`
                if given) and `X` is the transformed point set, shape
                `(nbatch, nsamples, dim)`.
        """
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
