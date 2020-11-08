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
from transformers import BertTokenizer, BertModel, BertConfig

# Modules
from .helpers import rdf_vocab_constructor
from models.decoder import Decoder

# How many triples to train and test system on (min: 1, max: 7)
MIN_NUM_TRIPLES = 1
MAX_NUM_TRIPLES = 1


def predict(x,
            word_embeddings,  # Decoder's word embeddings
            word2idx,  #
            idx2word,  #
            encoder,  #
            decoder,  #
            tokenizer,  #
            loss_fn,  #
            device,
            max_len=7,  #
            batch_size=32,  #
            compute_grads=False,  #
            targets=None,  #
            return_textual=False  # Whether to return predictions in index-form (default) or as textual strings
            ):
    print('In predict:')

    accumulated_loss = 0.

    # Init documentation of predictions
    predicted_indices = torch.zeros([batch_size, max_len]).to(device)  # Numeric
    if return_textual:
        predicted_words = [''] * batch_size

    # Tokenize sampled minibatch sentences
    inputs = tokenizer(x,
                       return_tensors="pt",  # Return tensors in pt=PyTorch format
                       padding=True,  # Pad all sentences in mini-batch to have the same length
                       add_special_tokens=True).to(device)  # Add "Start of sequence", "End of sequence", ... tokens.

    # print('Tokenized Inputs:', inputs)

    # Encode sentences: Pass tokenization output-dict-contents to model
    outputs = encoder(**inputs)
    # print('Got outputs:', outputs)

    # Retrieve hidden state to be passed into Decoder as annotation vectors
    # Reshape to get a set of 8 feature vectors from last hidden state
    annotations = outputs.last_hidden_state[:, -1, :].reshape(batch_size, 8, -1).to(device)
    # print('Annotations size after cropping & reshape:', annotations.size())

    # Init Decoder's hidden state
    hidden = decoder.init_hidden(annotations).unsqueeze(0).to(device)
    # print('Initial hidden size:', hidden.size(), 'given annotations:', annotations.size())

    # Construct initial embeddings (start tokens)
    embeddings = word_embeddings(torch.zeros([batch_size], dtype=int).to(device)).to(device)

    for t in range(max_len):
        # print('START OF ITERATION', t)
        # Get decodings (aka prob distrib. over output vocab per batch element) for time step t
        prob_dist, hidden = decoder(annotations,  # Static vector containing annotations per batch element
                                    embeddings,  # Word embedding predicted last iteration (per batch element)
                                    hidden  # Decoder's hidden state of last iteratipn per batch element
                                    )

        # Get predicted word index from predicted probability distribution (per batch element)
        word_index = torch.max(prob_dist, dim=1).indices
        # print('Predicted word indices batch:', word_index)

        # Get corresponding word embedding (by index; per batch element)
        embedding = word_embeddings(word_index.to(device))

        # TODO: optional teacher forcing?

        # Record predicted words
        predicted_indices[:, t] = word_index
        # print('Predicted indices:', predicted_indices)

        # Record textual words if required
        if return_textual:

            # Get predicted word (per batch element)
            predicted_word = [idx2word[batch_element.item()] for batch_element in word_index]

            for e in range(batch_size):
                predicted_words[e] += (predicted_word[e] + ' ')

        if compute_grads:
            # print('prob_dist:', prob_dist.size())
            # print('targets:', targets[:, t].size(), targets[:, t])

            # Compute (averaged over all batch elements given current time step t)
            loss = loss_fn(prob_dist, targets[:, t]).to(device)

            # Compute & back-propagate gradients
            loss.backward(retain_graph=True)

            # Document loss
            accumulated_loss += loss.item()
        # print('END OF ITERATION', t)

    ret_object = {
        'predicted_indices': predicted_indices,
    }

    print('Targets:\n', targets)
    print('Predicted idxs:\n', predicted_indices)

    if compute_grads:
        ret_object['loss'] = accumulated_loss
        # print('Accumulated loss:', accumulated_loss)
    if return_textual:
        ret_object['predicted_words'] = predicted_words
        # print("Predicted words:", predicted_words)
    # print("Returning from predict")
    return ret_object


def training(train_data,
             val_data,
             epochs,
             device,
             minibatch_size=32,
             embedding_dim=300,
             eval_frequency=10,  # Every how many epochs to run intermediate evaluation
             learning_rate_en=0.00001,
             learning_rate_de=0.00001,
             ):


    # Construct RDF vocab
    vocab_count, word2idx, idx2word = rdf_vocab_constructor(train_data)

    # Construct tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', device=device)

    # Construct embeddings
    rdf_vocab = nn.Embedding(num_embeddings=vocab_count, embedding_dim=embedding_dim, padding_idx=0).to(device)

    # Define model
    encoder = BertModel.from_pretrained('bert-base-uncased', return_dict=True).to(device)
    decoder = Decoder(
        annotation_size=(8, 96),  # Size of annotation vectors produced by Encoder
        out_vocab_size=vocab_count,  # How many words there are in the RDF-output language
        embedding_dim=300,  # Length of a word embedding
        hidden_dim=96,  # Nr hidden nodes
        output_dim=vocab_count,  # Vocab size
    ).to(device)

    # Optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate_en)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate_de)

    loss = nn.CrossEntropyLoss()

    # For both train and validation data & for all number of tuples per sentence
    # (in [MIN_NUM_TRIPLES, MAX_NUM_TRIPLES]), get the nr of train-/test instances
    len_x_train = [len(train_set) for train_set in train_data]
    len_x_val = [len(val_set) for val_set in val_data]

    # Development of both train- and validation losses over course of training
    train_losses, val_losses = [0.] * epochs, [0.] * epochs

    print('Starting training.')

    # Train
    for epoch in range(epochs):
        print('Epoch:', epoch)

        train_loss, eval_loss = 0., 0.

        # Reset gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Perform own train step for each nr of triples per sentence separately
        for i, nt in enumerate(range(MIN_NUM_TRIPLES, MAX_NUM_TRIPLES + 1)):
            print(str(i) + '. Condition:', nt, 'triples per sentence.')

            # Sample minibatch indices
            minibatch_idx = random.sample(population=range(len_x_train[i]), k=minibatch_size)
            # print('MB indices:', minibatch_idx)

            # Number of tokens to be predicted (per batch element)
            num_preds = nt * 3 + 1  # = nr triples * 3 + stop_token
            # print('Number of predictions:', num_preds)

            # Construct proper minibatch
            inputs = [train_data[i][idx]['text'] for idx in minibatch_idx]
            targets = torch.ones([minibatch_size, num_preds], dtype=int).to(device)

            # print('Inputs:', inputs)
            # print('Targets:', targets)

            for mb_i, idx in enumerate(minibatch_idx):
                # print('Text:', train_data[i][idx]['text'])
                # print('Triple:', train_data[i][idx]['triple'])
                for t, token in enumerate(train_data[i][idx]['triple']):
                    targets[mb_i, t] = word2idx[token]
            targets[:, -1] = 2  # 2 = Stop word index

            # print('Processed targets:', targets)
            # print('Predicting:')

            # Predict
            ret_object = predict(inputs,
                                 rdf_vocab,  # Decoder's word embeddings
                                 word2idx,  #
                                 idx2word,  #
                                 encoder,  #
                                 decoder,  #
                                 tokenizer,  #
                                 loss,  #
                                 device=device,
                                 max_len=num_preds,  # Nr of tokens to be predicted
                                 batch_size=32,  #
                                 compute_grads=True,  #
                                 targets=targets,  #
                                 return_textual=True
                                 # Whether to return predictions in index-form (default) or as textual strings
                                 )

            print('Return object:', ret_object)
            print("Predicted texts:", ret_object['predicted_words'])
            train_loss += ret_object['loss']
            # print("Returned loss:", ret_object['loss'])

        # Apply gradients
        encoder_optimizer.step()
        decoder_optimizer.step()
        # print('Optimizations performed.')

        # Intermediate evaluation

        # Save losses
        train_losses[epoch] = train_loss

    return train_losses, val_losses, encoder, decoder
