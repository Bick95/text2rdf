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
                embedding,      # Word embedding of most recently generated word (per batch element)
                h_old           # Previous hidden state (per batch element)
                ):

        # Note:
        # Unsqueeze/Squeeze operations done to add/remove extra 'sequence' dimension required for GRU layer;
        # There is no intrinsic sequence dimension in our data since we always only pass data through the GRU layer
        # for a single time step at a time before having to re-evaluate the attention weights.

        # Compute attention weights (1 per annotation vector (and per batch element))
        attention_weights = self.attn(annotations, h_old.squeeze(dim=0))

        # Weight individual annotation vectors by respective attention weights
        weighted_annotations = annotations * attention_weights

        # Compute weighted average annotation vector (=context vector) by summing over weighted annotation vectors
        context_vector = torch.sum(weighted_annotations, dim=1)

        # Append context vector (per batch element) by wordembedding predicted during last time step (per batch element)
        x = torch.cat((context_vector, embedding), dim=1)
        x = x.unsqueeze(1)

        # Feed context vector + wordembedding (per batch element) through GRU layer
        out, h = self.gru(x, h_old)
        out = out.squeeze(dim=1)

        # Computed probability distribution over output vocab (per batch element)
        prob_dist = self.softmax(self.fc(self.relu(out)))

        return prob_dist, h

    def init_hidden(self, annotation_vectors):
        # Mean annotation vector per batch element
        # Assumes that number of hidden nodes == number annotation features
        hidden = torch.mean(annotation_vectors, dim=1)  # .to(device)
        return hidden