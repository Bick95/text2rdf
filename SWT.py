#!/usr/bin/env python
# coding: utf-8

# Imports
# PyTorch
import torch

# Modules
from utils.train import training
from utils.dataset import get_train_vocab, get_dev_vocab, get_test_vocab

# CUDA related
device = torch.device('cuda')
print("Device:", device)


def main():

    # Free CUDA memory
    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    # Get datasets
    train, train_stats = get_train_vocab()
    test, test_stats = get_test_vocab()
    train, train_stats, dev, dev_stats = get_dev_vocab(train, train_stats)

    # Train
    train_losses, val_losses, encoder, decoder = training(train, dev, device=device, epochs=1000)
    print('Train losses:', train_losses)


if __name__ == "__main__":
    # execute only if run as a script
    main()
