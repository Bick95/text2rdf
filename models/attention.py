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
        # print('SA INIT')
        # Variables
        self.num_annotations = annotation_size[0]
        self.annotation_features = annotation_size[1]
        self.hidden_size = hidden_len

        # Layers
        self.attn = nn.Linear(self.annotation_features + self.hidden_size, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, annotations, prev_hidden):
        # Repeat prev_hidded X times to append it to each of the annotation vectors (per batch element)
        print('In SA forward...', prev_hidden.size())
        repeated_hidden = torch.cat(
            [
                torch.repeat_interleave(hid, repeats=self.num_annotations, dim=0).unsqueeze(0)
                for hid in prev_hidden.split(1)
            ]
        )

        # Append previous hidden state to all annotation vectors (for each individual batch element)
        # Input to attention weight calculation
        input = torch.cat((annotations, repeated_hidden), dim=2)

        # Compute the relative attention scores per feaure (e_{ti}=f_{att}(a_i,h_{t−1}) from paper)
        energies = self.attn(input)

        # Compute final attention weights (i.e. alpha)
        attn_weights = self.softmax(energies)

        return attn_weights