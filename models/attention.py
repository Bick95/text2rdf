#!/usr/bin/env python
# coding: utf-8

# Imports
# PyTorch
import torch
import torch.nn as nn


class SoftAttention(nn.Module):

    def __init__(self,
                 annotation_size,  # Tuple: (num_annotations, num_features_per_annotation)
                 hidden_len  # Number of nodes in Decoder's hidden state weight matrix
                 ):
        super(SoftAttention, self).__init__()

        # Soft Attention module implemented following: http://proceedings.mlr.press/v37/xuc15.pdf

        # Variables
        self.num_annotations = annotation_size[0]       # Nr of annotation vectors (per input sentence and batch element)
        self.annotation_features = annotation_size[1]   # Nr of features per annotation vector
        self.hidden_size = hidden_len                   # How many nodes there are in Decoder's hidden state

        # Layers
        self.attn = nn.Linear(self.annotation_features + self.hidden_size, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, annotations, prev_hidden):
        # Repeat Decoder's previous hidden state X times to append it to each annotation vector below (per batch element)
        repeated_hidden = torch.cat(
            [
                torch.repeat_interleave(hid, repeats=self.num_annotations, dim=0).unsqueeze(0)
                for hid in prev_hidden.split(1)
            ]
        )

        # Append previous hidden state to all annotation vectors (for each individual batch element)
        input = torch.cat((annotations, repeated_hidden), dim=2)

        # Compute the relative attention scores per annotation feature (e_{ti}=f_{att}(a_i,h_{t−1}) from paper)
        energies = self.attn(input)

        # Compute final attention weights (i.e. variable 'alpha' in the paper)
        attn_weights = self.softmax(energies)

        return attn_weights