#!/usr/bin/env python
# coding: utf-8

# Imports
# PyTorch
import torch
import torch.nn as nn

# Modules
from .attention import SoftAttention


class Decoder(nn.Module):

    def __init__(self,
                 annotation_size,  # Size of annotation vectors produced by Encoder
                 embedding_dim,  # Length of a word embedding
                 hidden_dim,  # Nr hidden nodes
                 output_dim,  # Vocab size (How many words there are in the RDF-output language)
                 bidirectional=False,  # Whether to have bidirectional GRU
                 n_layers=1,  # Nr layers in GRU
                 drop_prob=0.2  # Percent of node-dropouts
                 ):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_directions = 1 if not bidirectional else 2  # TODO: make use of it...

        self.attn = SoftAttention(annotation_size=annotation_size, hidden_len=hidden_dim)
        self.gru = nn.GRU(annotation_size[1] + embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,
                annotations,    # Static annotation vectors (for each batch element)
                embeddings,     # Word embeddings of most recently generated word (per batch element)
                h_old           # Previous hidden state per batch element
                ):
        annotation_weights = self.attn(annotations, h_old.squeeze(dim=0))
        weighted_annotations = annotations * annotation_weights
        context_vectors = torch.sum(weighted_annotations, dim=1)

        x = torch.cat((context_vectors, embeddings), dim=1)
        x = x.unsqueeze(1)  # Add une dimension for 'sequence'

        out, h = self.gru(x, h_old)
        out = out.squeeze(dim=1)
        out = self.softmax(self.fc(self.relu(out)))

        return out, h

    def init_hidden(self, annotation_vectors):
        # Mean of annotation vector per batch element
        # Assumes that number of hidden nodes == number annotation features
        hidden = torch.mean(annotation_vectors, dim=1)  # .to(device)
        return hidden