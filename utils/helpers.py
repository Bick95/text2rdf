#!/usr/bin/env python
# coding: utf-8

# Imports
# Bleu score
from nltk.translate.bleu_score import corpus_bleu


def rdf_vocab_constructor(raw_vocab):
    vocab_count, word2idx, idx2word = 3, {'START': 0, 'PAD': 1, 'END': 2}, {0: 'START', 1: 'PAD', 2: 'END'}

    for partition in raw_vocab:  # Different partitions with respect to nr or triples per sentence
        for train_instance in partition:
            triple = train_instance['triple']
            for token in triple:
                if token not in word2idx:
                    word2idx[token] = vocab_count
                    idx2word[vocab_count] = token
                    vocab_count += 1
    return vocab_count, word2idx, idx2word


# Function for calculating the BLEU score for multiple sentence.
def calculate_bleu(data, train, dev, test, model, max_len=7):
    trgs = []
    pred_trgs = []
    src = dev
    trg = test
    # Get the data and feed it into pred_trg after Seq2seq
    # pred_trg = pred_trg[:-1]
    # pred_trgs.append(pred_trg)
    # trgs.append([trg])

    return corpus_bleu(pred_trgs, trgs)

