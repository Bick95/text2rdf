#!/usr/bin/env python
# coding: utf-8

# Imports
# General purpose
import random

# PyTorch
import torch


def predict(x,                # Batch of input sentences
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
