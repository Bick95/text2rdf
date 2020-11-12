import torch


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


def count_matches(targets, predictions, conf_matrix, debug=False):

    # This function takes target triples (targets) to be extracted from a
    # single input sentence and the corresponding predicted triples created
    # for said sentence. Also, it takes a confusion-matrix to encode true
    # positives, false positives, and false negatives per triple.
    # It returns the updated confusion-matrix dictionary.

    if debug:
        print('Targets:\n', targets)
        print('Preds:\n', predictions)

    for target_ in targets:
        target = str(target_)
        # print('Target:', target)
        # target is a triple
        if target_ in predictions:
            # Hit! Triple correctly extracted from sentence
            if debug:
                print('Hit:\t', target)

            # Record hit (=true positive)
            if target in conf_matrix:
                # Triple is part of dictionary already, increment its hit/true-positive-count
                conf_matrix[target]['TP'] += 1
                conf_matrix[target]['is_in_targets'] = True  # Make sure it's marked as part of reference solution
            else:
                # First time that target triple is encountered - add it to conf-matrix
                conf_matrix[target] = {
                    'TP': 1,
                    'FP': 0,
                    'FN': 0,
                    'is_in_targets': True
                }

            # Don't match same token twice
            predictions.remove(target_)

        else:
            # Target has not been predicted, it's a miss (=false negative)
            if debug:
                print('Miss:\t', target)

            # Record miss (=false negative)
            if target in conf_matrix:
                # Triple is part of dictionary already, increment its hit/true-positive-count
                conf_matrix[target]['FN'] += 1
                conf_matrix[target]['is_in_targets'] = True  # Make sure it's marked as part of reference solution
            else:
                # First time that target triple is encountered - add it to conf-matrix
                conf_matrix[target] = {
                    'TP': 0,
                    'FP': 0,
                    'FN': 1,
                    'is_in_targets': True  # It has not been observed as a target yet
                }

    if debug:
        print('Excesses:\t', predictions)

    for leftover_ in predictions:
        leftover = str(leftover_)
        # Triple has been predicted incorrectly - it's not part of reference solution for given sentence
        if leftover in conf_matrix:
            # Triple is part of dictionary already, increment its false-positive-count
            conf_matrix[leftover]['FP'] += 1
        else:
            # First time that predicted triple is encountered - add it to conf-matrix
            conf_matrix[leftover] = {
                'TP': 0,
                'FP': 1,
                'FN': 0,
                'is_in_targets': False
            }

    return conf_matrix


def evaluation(
        val_data,
        rdf_vocab,  # Decoder's word embeddings
        word2idx,
        idx2word,
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

    # Returns average TP, FP, and FN (computed over all triples contained
    # in target set) and confusion matrix.

    # Data structure that has triples as vocab and records mis|matches per triple (variant of confusion matrix)
    conf_matrix = {
        # 'Dog eats food': { tp: 1,  - Correctly predicted this triple
        #                    fp: 0,  - Erroneously predicted this triple
        #                    fn: 0,  - Miss to predict this triple
        #                    is_in_targets: true - Is triple contained in targets or made up by decoder?
        #                   }
    }

    # Perform eval steps for each nr of triples per sentence separately
    for nt in range(min_nr_triples, max_nr_triples + 1):
        i = nt - min_nr_triples  # Num triples per sentence starts at 1, while indexing starts at 0
        # print(min_nr_triples, max_nr_triples, nt, i, len_x_val)
        for element_idx in range(len_x_val[i]):
            # print('Element idx:', str(element_idx) + ',', str(i) + '. Condition:', nt, 'triples per sentence.')

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
                print('Targets sentences:', [[idx2word[index] for index in lst] for lst in target_triples])

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

            if debug:
                print('Pred triples: ', pred_triples)
                print('Predictions:', [[idx2word[index] for index in lst] for lst in pred_triples])

            conf_matrix = count_matches(target_triples, pred_triples, conf_matrix, debug=debug)

    # Compute summed TP, FP, and FN
    tp, fp, fn, cnt_targets, cnt_made_up = 0., 0., 0., 0., 0.

    print(conf_matrix)

    for triple_key in conf_matrix:
        obj = conf_matrix[triple_key]
        if obj['is_in_targets']:
            tp += obj['TP']
            fp += obj['FP']
            fn += obj['FN']
            cnt_targets += 1
        else:
            cnt_made_up += 1

    # Summaries|normalizations could be computed later
    # tp /= cnt_targets
    # fp /= cnt_targets
    # fn /= cnt_targets

    return tp, fp, fn, cnt_targets, cnt_made_up, conf_matrix

