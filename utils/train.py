#!/usr/bin/env python
# coding: utf-8

# Imports
# General purpose
import random

# PyTorch
import torch
import torch.nn as nn
from torch import optim

# Bert
from transformers import BertTokenizer, BertModel

# Modules
from .helpers import rdf_vocab_constructor, get_max_sentence_len
from .evaluate import inference, evaluate
from models.decoder import Decoder
from .predict import predict


def training(train_data,
             val_data,
             minibatch_updates,
             device,
             minibatch_size=32,
             embedding_dim=300,
             eval_frequency=10,  # Every how many minibatch_updates to run intermediate evaluation
             learning_rate_en=0.00001,
             learning_rate_de=0.0001,
             teacher_forcing_max=1.,
             teacher_forcing_min=0.,
             teacher_forcing_dec=0.05,
             min_nr_triples=1,
             max_nr_triples=3
             ):


    # Construct RDF vocab
    vocab_count, word2idx, idx2word = rdf_vocab_constructor(train_data)

    # Construct tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', device=device)

    # Get max len of among sentences
    max_sen_len = get_max_sentence_len(train_data,
                                       tokenizer=tokenizer,
                                       min_nr_triples=min_nr_triples,
                                       max_nr_triples=max_nr_triples)
    print("Max sentence len is:", max_sen_len)

    # Construct embeddings
    rdf_vocab = nn.Embedding(num_embeddings=vocab_count, embedding_dim=embedding_dim, padding_idx=0).to(device)

    # Define model
    encoder = BertModel.from_pretrained('bert-base-uncased', return_dict=True).to(device)
    decoder = Decoder(
        annotation_size=(max_sen_len, 768),  # Size of annotation vectors produced by Encoder
        embedding_dim=300,  # Length of a word embedding
        hidden_dim=768,  # Nr hidden nodes
        output_dim=vocab_count,  # Vocab size
    ).to(device)

    teacher_forcing = teacher_forcing_max

    # Optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate_en)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate_de)

    loss = nn.NLLLoss(reduction='mean').cuda()

    # For both train and validation data & for all number of tuples per sentence
    # (in [MIN_NUM_TRIPLES, MAX_NUM_TRIPLES]), get the nr of train-/test instances
    len_x_train = [len(train_set) for train_set in train_data]
    len_x_val = [len(val_set) for val_set in val_data]

    # Development of both train- and validation losses over course of training
    train_losses, val_losses = [0.] * minibatch_updates, [0.] * minibatch_updates

    print('Starting training.')

    # Train
    for epoch in range(minibatch_updates):

        train_loss, eval_loss = 0., 0.

        # Reset gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Perform own train step for each nr of triples per sentence separately
        for nt in range(min_nr_triples, max_nr_triples + 1):
            i = nt - 1  # Num triples per sentence starts at 1, while indexing starts at 0
            print('Epoch:', str(epoch) + ',', str(i) + '. Condition:', nt, 'triples per sentence.')

            # Sample minibatch indices
            minibatch_idx = random.sample(population=range(len_x_train[i]), k=minibatch_size)

            # Number of tokens to be predicted (per batch element)
            num_preds = nt * 3 + 1  # = nr triples * 3 + stop_token

            # Construct proper minibatch
            inputs = [train_data[i][idx]['text'] for idx in minibatch_idx]
            targets = torch.ones([minibatch_size, num_preds], dtype=int).to(device)

            for mb_i, idx in enumerate(minibatch_idx):
                for t, token in enumerate(train_data[i][idx]['triple']):
                    targets[mb_i, t] = word2idx[token]
            targets[:, -1] = 2  # 2 = Stop word index

            if epoch % eval_frequency is 0:
                # Print trainable parameter stats
                for name, param in decoder.named_parameters():
                    if param.requires_grad:
                        print('Params - Name:', name,
                              'mean:', torch.mean(param.data),
                              'max:', torch.max(param.data),
                              'min:', torch.min(param.data))
                # Print input sentences
                print('Input sentences:\n', inputs)
                return_textual = True
            else:
                return_textual = False

            # Predict
            ret_object = predict(inputs,
                                 rdf_vocab,  # Decoder's word embeddings
                                 word2idx,
                                 idx2word,
                                 encoder,
                                 decoder,
                                 tokenizer,
                                 loss,
                                 device=device,
                                 max_len=num_preds,  # Nr of tokens to be predicted
                                 max_sen_len=max_sen_len,
                                 batch_size=32,
                                 compute_grads=True,
                                 targets=targets,
                                 return_textual=return_textual,
                                 teacher_forcing=teacher_forcing
                                 )

            train_loss += ret_object['loss']
            print("Iteration loss:", ret_object['loss'])

            if epoch % eval_frequency is 0:
                print('Percent correctly predicted:',
                      (torch.eq(targets, ret_object['predicted_indices']).sum() / (minibatch_size * num_preds)).item())

        # Apply gradients
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Intermediate evaluation
        if epoch % eval_frequency == 0:
            y_true, y_pred = inference(val_data, encoder, decoder, rdf_vocab, word2idx, idx2word, device, tokenizer)
            metrics = evaluate(y_true, y_pred)
            print('Metrics:', metrics)

        # Save losses
        train_losses[epoch] = train_loss

        teacher_forcing = max(teacher_forcing_min, teacher_forcing-teacher_forcing_dec)

    return train_losses, val_losses, encoder, decoder, word2idx, idx2word
