# Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation:
# https://papers.nips.cc/paper/7877-graph-convolutional-policy-network-for-goal-directed-molecular-graph-generation.pdf

from .embedmodel import *
from .data import *

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets.utils import download_url

from rdkit import Chem, RDLogger
from rdkit.Chem import RDConfig, Descriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.Scaffolds import MurckoScaffold

import copy
from collections import defaultdict, deque
from collections.abc import Sequence

import networkx as nx
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")
np.random.seed(42)

class ReusableLayers(nn.Module):
 
    def __init__(self, input_dim, hidden_dims, short=False, batch_norm=False, activation="relu", dropout=0):
        super(ReusableLayers, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.dims = [input_dim] + hidden_dims
        self.short = short

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
        if batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(self.dims) - 2):
                self.batch_norms.append(nn.BatchNorm1d(self.dims[i + 1]))
        else:
            self.batch_norms = None

    def forward(self, input):

        layer_input = input

        for i, layer in enumerate(self.layers):
            hidden = layer(layer_input)
            if i < len(self.layers) - 1:
                if self.batch_norms:
                    x = hidden.flatten(0, -2)
                    hidden = self.batch_norms[i](x).view_as(hidden)
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
            if self.short and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            layer_input = hidden

        return hidden

class GCPN(nn.Module):
    """ 
    TODO:
        graph representation model created outside
        set of all possible atom types
        property optimization 
        max number of node.
        ppo implementation (update, gamma etc.)
    """

    def __init__(self, model, atom_types, max_node=None,
                 hidden_dim_mlp=128, update_interval=10, gamma=0.9, baseline_momentum=0.9):
        super(GCPN, self).__init__()
        self.model = model
        self.max_node = max_node
        self.hidden_dim_mlp = hidden_dim_mlp
        self.update_interval = update_interval
        self.gamma = gamma
        self.baseline_momentum = baseline_momentum
        self.best_results = defaultdict(list)
        self.batch_id = 0             

        self.embedings = nn.Parameter(torch.zeros(self.id2atom.size(0), self.model.output_dim))
        nn.init.normal_(self.embedings, mean=0, std=0.1)
        self.inp_dim_stop = self.model.output_dim
        self.mlp_stop = ReusableLayers(self.inp_dim_stop, [self.hidden_dim_mlp, 2], activation='tanh')   

        self.inp_dim_node1 = self.model.output_dim + self.model.output_dim
        self.mlp_node1 = ReusableLayers(self.inp_dim_node1, [self.hidden_dim_mlp, 1], activation='tanh')
        self.inp_dim_node2 = 2 * self.model.output_dim + self.model.output_dim
        self.mlp_node2 = ReusableLayers(self.inp_dim_node2, [self.hidden_dim_mlp, 1], activation='tanh')
        self.inp_dim_edge = 2 * self.model.output_dim
        self.mlp_edge = ReusableLayers(self.inp_dim_edge, [self.hidden_dim_mlp, self.model.num_relation], activation='tanh')

        self.agent_model = copy.deepcopy(self.model)
        self.agent_embedings = copy.deepcopy(self.embedings)
        self.agent_mlp_stop = copy.deepcopy(self.mlp_stop)
        self.agent_mlp_node1 = copy.deepcopy(self.mlp_node1)
        self.agent_mlp_node2 = copy.deepcopy(self.mlp_node2)
        self.agent_mlp_edge = copy.deepcopy(self.mlp_edge)

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        loss, metric = self.reinforce(batch)
        all_loss += loss 
        metric.update(metric)
        return all_loss, metric

    def predict(self, graph, label_dict, use_agent=False):
        # step1: get node/graph embeddings
            # (num_graph * 16)  # nodeamount * allAtomtypes
            # (num_node + 16 * num_graph)  # graphID_foreachnode + nodeamount*allAtomtypes
            # [moleculeGraph_feature_d][nodeamount * allAtomtypes+1]-->for each node as graphID
        # step2: predict stop
            #(num_graph, n_out)
            #(num_graph, 2)
            #(num_graph, 2)
        # step3: predict first node: node1
            #(num_node, n_out)
            # (num_node + 16 * num_graph, n_out)
            # (num_node + 16 * num_graph, n_out)
            # cat graph emb
            #(num_node + 16 * num_graph)
            #(num_node + 16 * num_graph)
            #mask the extended part

        # step4: predict second node: node2      
            #mask the selected node1

        # step5: predict edge type

            # (num_graph)
            #(num_graph, n_out)
            #(num_graph, n_out)
            #(num_graph, 2n_out)
            # (num_graph, num_relation)
            # (num_graph, num_relation)

        #return stop_logits, node1_logits, node2_logits, edge_logits, index_dict
        return


    def reinforce(self, batch):
        # rewards
        # per graph size reward baseline
        # return all losses and metrics
        return 