#!/usr/bin/env python
# coding: utf-8

# Imports
# Saving
import json
import random
from datetime import datetime

# PyTorch
import torch

# Modules
from utils.train import training, precision, recall
from utils.evaluate import evaluation
from utils.dataset import get_train_vocab, get_dev_vocab, get_test_vocab, print_stats

# CUDA related
device = torch.device('cuda')
print("Device:", device)


# How many triples to train and test system on (min: 1, max: 7)
MIN_NUM_TRIPLES = 1
MAX_NUM_TRIPLES = 2

MINIBATCH_UPDATES = 60


def main():

    # Free CUDA memory
    if str(device) == 'cuda':
        torch.cuda.empty_cache()

    eval_data = {
        'min_num_triples': MIN_NUM_TRIPLES,
        'max_num_triples': MAX_NUM_TRIPLES,
        'minibatch_updates': MINIBATCH_UPDATES,
        # 'train_losses': train_losses,
        'val': {
        #    epoch: {
        #        'TP': ...,
        #        'FP': ...,
        #        'FN': ...,
        #        'prec': ...,
        #        'rec': ...
        #    },
        },
        #'test': {
        #    'TP': ...,
        #    'FP': ...,
        #    'FN': ...,
        #    'prec': ...,
        #    'rec': ...
        #}
    }

    # Get datasets
    train, train_stats = get_train_vocab(min_num_triples=MIN_NUM_TRIPLES, max_num_triples=MAX_NUM_TRIPLES)
    test, test_stats = get_test_vocab(min_num_triples=MIN_NUM_TRIPLES, max_num_triples=MAX_NUM_TRIPLES)
    train, train_stats, dev, dev_stats = get_dev_vocab(train, train_stats,
                                                       min_num_triples=MIN_NUM_TRIPLES, max_num_triples=MAX_NUM_TRIPLES)

    print_stats(train, dev, test, min_num_triples=MIN_NUM_TRIPLES, max_num_triples=MAX_NUM_TRIPLES)

    # Train
    eval_data, encoder, decoder, word2idx, idx2word, rdf_vocab,\
        tokenizer, max_sen_len = training(train,
                                          dev,
                                          eval_data,
                                          device=device,
                                          minibatch_updates=MINIBATCH_UPDATES,
                                          min_nr_triples=MIN_NUM_TRIPLES,
                                          max_nr_triples=MAX_NUM_TRIPLES
                                         )
    print('Train losses:', eval_data['train_losses'])

    # For test data & for all number of tuples per sentence
    # (in [MIN_NUM_TRIPLES, MAX_NUM_TRIPLES]), get the nr of train-/test instances
    len_x_test = [len(test_set) for test_set in test]

    tp, fp, fn, conf_matrix = evaluation(
        test,
        rdf_vocab,  # Decoder's word embeddings
        word2idx,
        idx2word,
        device,
        encoder,
        decoder,
        tokenizer,
        len_x_test,
        max_sen_len,
        min_nr_triples=MIN_NUM_TRIPLES,
        max_nr_triples=MAX_NUM_TRIPLES,
        end_token_idx=word2idx['END'],
        max_pred_len=30,
        debug=True
    )
    print('Final eval:')
    print('Conf matrix:', conf_matrix)
    print('TP:', tp, 'FP:', fp, 'FN:', fn)

    # Save test stats
    eval_data['test'] = {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'prec': precision(tp, fp),
        'rec': recall(tp, fn)
    }

    # Save eval_data object in name with unique name
    name = 'eval_data_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '_' + str(random.randint(0, 999999)) + '.txt'
    with open(name, 'w') as outfile:
        json.dump(eval_data, outfile)


if __name__ == "__main__":
    # execute only if run as a script
    main()
