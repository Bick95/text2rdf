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


def get_max_sentence_len(raw_vocab, tokenizer, min_nr_triples=1, max_nr_triples=1):
    longest_len = 0
    for nt in range(min_nr_triples-1, max_nr_triples):
        print("Checking nr of triples:", nt+1, '(for len)')
        for entry in raw_vocab[nt]:
            l = tokenizer([entry['text']],
                               return_tensors="pt",  # Return tensors in pt=PyTorch format
                               padding=False,  # Pad all sentences in mini-batch to have the same length
                               add_special_tokens=True)['input_ids'].size()[1]
            if l > longest_len:
                print('Longer sentence of len', l, ':', entry['text'])
                print('Corresponding triple:', entry['triple'])
                longest_len = l
    return longest_len


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

