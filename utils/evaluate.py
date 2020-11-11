import copy
import torch
from sklearn import metrics as skmetrics

from .predict import predict


def predict_singleton(x,
                      word_embeddings,
                      encoder,
                      decoder,
                      tokenizer,
                      device,
                      max_pred_len=30,
                      max_sen_len=66,
                      end_token_idx=2
                      ):

    # Predicted output tokens
    predicted_indices = []

    # Tokenize sampled minibatch sentences
    inputs = tokenizer(x,
                       return_tensors="pt",  # Return tensors in pt=PyTorch format
                       padding=True,  # Pad all sentences in mini-batch to have the same length
                       add_special_tokens=True).to(device)  # Add "Start of sequence", "End of sequence", ... tokens.

    # Encode sentences: Pass tokenization output-dict-contents to model
    outputs = encoder(**inputs)

    # Retrieve hidden state to be passed into Decoder as annotation vectors
    # 0-padded vector containing a 768-dimensional feature representation per word
    annotations = outputs.last_hidden_state
    annotations_padded = torch.zeros([1, max_sen_len, 768], dtype=torch.float32).to(device)
    s = annotations.size()
    annotations_padded[:s[0], :s[1], :s[2]] = annotations[:, :max_sen_len, :]  # Crop if sentence is longer than max sentence len

    # Init Decoder's hidden state
    hidden = decoder.init_hidden(annotations_padded).unsqueeze(0).to(device)

    # Construct initial embedding (start tokens) per batch element
    embedding = word_embeddings(torch.zeros([1], dtype=int).to(device)).to(device)

    for t in range(max_pred_len):
        # Get decodings (aka prob distrib. over output vocab per batch element) for time step t
        prob_dist, hidden = decoder(annotations_padded,  # Static vector containing annotations per batch element
                                    embedding,  # Word embedding predicted last iteration (per batch element)
                                    hidden  # Decoder's hidden state of last iteratipn per batch element
                                    )

        # Get predicted word index from predicted probability distribution (per batch element)
        word_index = torch.max(prob_dist, dim=1).indices

        # Get corresponding, predicted word embedding (by index; per batch element)
        embedding = word_embeddings(word_index.to(device))

        # Record predicted words
        predicted_indices.append(word_index.item())

        if word_index.item() == end_token_idx:
            break

    return predicted_indices


def count_matches(targets, predictions, debug=False):
    tp, fp, fn = 0, 0, 0  # tp=correctly predicted == hit,
                          # fp=prediction not part of targets,
                          # fn=absence of target token in predictions == miss
    if debug:
        print('Targets:\n', targets)
        print('Preds:\n', predictions)

    for e in targets:
        if e in predictions:
            if debug:
                print('Hit:\t', e)
            tp += 1
            predictions.remove(e)
        else:
            if debug:
                print('Miss:\t', e)
            fn += 1

    fp = len(predictions)  # Leftover; if leftover had been part of predictions, it would not be in this list any longer
    if debug:
        print('Excesses:\t', predictions)
    return tp, fp, fn


def evaluation(
        val_data,
        rdf_vocab,  # Decoder's word embeddings
        word2idx,
        device,
        encoder,
        decoder,
        tokenizer,
        len_x_val,
        max_sen_len,
        min_nr_triples=1,
        max_nr_triples=3,
        end_token_idx=2,
        max_pred_len=30,
        debug=False
     ):

    hits, excesses, misses, true_neg = [0] * len(len_x_val), [0] * len(len_x_val), [0] * len(len_x_val), [0] * len(len_x_val)

    # Perform eval steps for each nr of triples per sentence separately
    for nt in range(min_nr_triples, max_nr_triples + 1):
        i = nt - min_nr_triples  # Num triples per sentence starts at 1, while indexing starts at 0
        print(min_nr_triples, max_nr_triples, nt, i, len_x_val)
        for element_idx in range(len_x_val[i]):
            print('Element idx:', str(element_idx) + ',', str(i) + '. Condition:', nt, 'triples per sentence.')

            # Construct pseudo-minibatch
            inputs = [val_data[i][element_idx]['text']]

            # Get indices of words
            target_triples = [word2idx[x] if x in word2idx else 1 for x in val_data[i][element_idx]['triple']] + [word2idx['END']]

            # Get all contained triples in terms of indices
            target_triples = [target_triples[s:s+3] for s in range(0, len(target_triples), 3)]

            if debug:
                # Print input sentences & triples
                print('Input sentences:\n', inputs)
                print('Target triples:\n', target_triples)

            # Predict
            predict_indices = predict_singleton(
                x=inputs,
                word_embeddings=rdf_vocab,
                encoder=encoder,
                decoder=decoder,
                tokenizer=tokenizer,
                device=device,
                max_pred_len=max_pred_len,
                max_sen_len=max_sen_len,
                end_token_idx=end_token_idx
            )

            # Get all contained triples in terms of indices
            pred_triples = [predict_indices[s:s + 3] for s in range(0, len(predict_indices), 3)]

            tp, fp, fn = count_matches(target_triples, pred_triples, debug=debug)

            hits[i] += tp
            excesses[i] += fp
            misses[i] += fn

    return hits, excesses, misses, true_neg

