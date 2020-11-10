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
from .helpers import rdf_vocab_constructor, get_max_sentence_len
from models.decoder import Decoder


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
            max_sen_len=66,
            batch_size=32,  #
            teacher_forcing=0.,
            compute_grads=False,  #
            targets=None,  #
            return_textual=False  # Whether to return predictions in index-form (default) or as textual strings
            ):

    if teacher_forcing:
        # Stochastically determine whether to apply teacher forcing in current iteration
        teacher_forcing = random.random() < teacher_forcing

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

    # Retrieve hidden state to be passed into Decoder as annotation vectors
    # 0-padded vector containing a 768-dimensional feature representation per word
    annotations = outputs.last_hidden_state
    annotations_padded = torch.zeros([batch_size, max_sen_len, 768], dtype=torch.float32).to(device)
    s = annotations.size()
    annotations_padded[:s[0], :s[1], :s[2]] = annotations

    # Init Decoder's hidden state
    hidden = decoder.init_hidden(annotations_padded).unsqueeze(0).to(device)

    # Construct initial embedding (start tokens) per batch element
    embedding = word_embeddings(torch.zeros([batch_size], dtype=int).to(device)).to(device)

    for t in range(max_len):
        # Get decodings (aka prob distrib. over output vocab per batch element) for time step t
        prob_dist, hidden = decoder(annotations_padded,  # Static vector containing annotations per batch element
                                    embedding,  # Word embedding predicted last iteration (per batch element)
                                    hidden  # Decoder's hidden state of last iteratipn per batch element
                                    )

        # Get predicted word index from predicted probability distribution (per batch element)
        word_index = torch.max(prob_dist, dim=1).indices

        if teacher_forcing:
            # Apply teacher forcing & pretend network had predicted correct tokens previously
            embedding = word_embeddings(targets[:, t])

        else:
            # Get corresponding, predicted word embedding (by index; per batch element)
            embedding = word_embeddings(word_index.to(device))


        # Record predicted words
        predicted_indices[:, t] = word_index

        # Record textual words if required
        if return_textual:

            # Get predicted word (per batch element)
            predicted_word = [idx2word[batch_element.item()] for batch_element in word_index]

            for e in range(batch_size):
                predicted_words[e] += (predicted_word[e] + ' ')

        if compute_grads:

            # Compute (averaged over all batch elements given current time step t)
            loss = loss_fn(torch.log(prob_dist), targets[:, t]).to(device)

            # Retain computational graph through all but last backward pass
            retain_graph = not t == max_len-1

            # Compute & back-propagate gradients
            loss.backward(retain_graph=retain_graph)

            # Document loss
            accumulated_loss += loss.detach().item()

    ret_object = {
        'predicted_indices': predicted_indices,
    }

    if compute_grads:
        ret_object['loss'] = accumulated_loss
    if return_textual:
        ret_object['predicted_words'] = predicted_words
        print("Predicted words:\n", predicted_words)
        print('Targets:\n', targets)

        print('Predicted idxs:\n', predicted_indices)

    return ret_object


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
             teacher_forcing_min=0.1,
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

        # Apply gradients
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Intermediate evaluation

        # Save losses
        train_losses[epoch] = train_loss

        teacher_forcing = max(teacher_forcing_min, teacher_forcing-teacher_forcing_dec)

    return train_losses, val_losses, encoder, decoder
