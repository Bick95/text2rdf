#!/usr/bin/env python
# coding: utf-8

# Imports
# PyTorch
import torch

# Modules
from utils.train import training
from utils.evaluate import inference
from utils.dataset import get_train_vocab, get_dev_vocab, get_test_vocab, print_stats

# CUDA related
device = torch.device('gpu')
print("Device:", device)


# How many triples to train and test system on (min: 1, max: 7)
MIN_NUM_TRIPLES = 1
MAX_NUM_TRIPLES = 2

MINIBATCH_UPDATES = 50


def main():

    # Free CUDA memory
    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    # Get datasets
    train, train_stats = get_train_vocab(min_num_triples=MIN_NUM_TRIPLES, max_num_triples=MAX_NUM_TRIPLES)
    test, test_stats = get_test_vocab(min_num_triples=MIN_NUM_TRIPLES, max_num_triples=MAX_NUM_TRIPLES)
    train, train_stats, dev, dev_stats = get_dev_vocab(train, train_stats,
                                                       min_num_triples=MIN_NUM_TRIPLES, max_num_triples=MAX_NUM_TRIPLES)

    print_stats(train, dev, test, min_num_triples=MIN_NUM_TRIPLES, max_num_triples=MAX_NUM_TRIPLES)

    # Train
    train_losses, val_losses, encoder, decoder, word2idx, idx2word= training(train,
                                                                            dev,
                                                                            device=device,
                                                                            minibatch_updates=MINIBATCH_UPDATES,
                                                                            min_nr_triples=MIN_NUM_TRIPLES,
                                                                            max_nr_triples=MAX_NUM_TRIPLES
                                                                            )
    print('Train losses:', train_losses)

    if str(device) == 'gpu':
        torch.cuda.empty_cache()

    y_true, y_pred = inference(dev, encoder, decoder, rdf_vocab, word2idx, idx2word)
    metrics = evaluate(y_true, y_pred)
    print(metrics)


if __name__ == "__main__":
    # execute only if run as a script
    main()
