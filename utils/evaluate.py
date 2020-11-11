import copy
import torch
from sklearn import metrics as skmetrics

from .train import predict


def inference(
        data,
        encoder, decoder,
        rdf_vocab, word2idx, idx2word,
        device,
        tokenizer,
        embedding_dim=300,
        minibatch_size=20,
        min_num_triples=1,
        max_num_triples=3,
    ):
    
    # Construct RDF vocab
    # vocab_count, word2idx, idx2word = rdf_vocab_constructor(data)
    len_x = [len(dataset) for dataset in data]

    # Perform own train step for each nr of triples per sentence separately
    for i, nt in enumerate(range(min_num_triples, max_num_triples+1)):
        print(str(i) + '. Condition:', nt, 'triples per sentence.')
        
        y_pred = []
        y_true = []
        
        for batch in range(0, len_x[i], minibatch_size):

            batch_end = min(len(data[i]), batch+minibatch_size)
            
            # Number of tokens to be predicted (per batch element)
            num_preds = nt*3+1 # = nr triples * 3 + stop_token 

            # Construct proper minibatch
            inputs = [x['text'] for x in data[i][batch:batch_end]]
            targets = torch.ones([minibatch_size, num_preds], dtype=int).to(device)

            for mb_i, idx in enumerate(range(batch, batch_end)):
                for t, token in enumerate(data[i][idx]['triple']):
                    targets[mb_i, t] = word2idx[token] if token in word2idx else 1
            targets[:, -1] = 2  # 2 = Stop word index
            
            targets_text = [tuple(x['triple']) for x in data[i][batch:batch+minibatch_size]]

            # Predict
            ret_object = predict(
                inputs,
                rdf_vocab,              # Decoder's word embeddings
                word2idx,               # 
                idx2word,               # 
                encoder,                # 
                decoder,                # 
                tokenizer,              # 
                loss_fn=None,           #
                device=device,          #
                max_len=num_preds,      # Nr of tokens to be predicted
                batch_size=batch_end - batch,          # 
                compute_grads=False,     # 
                targets=targets,        # 
                return_textual=False     # Whether to return predictions in index-form (default) or as textual strings
            )
            
            y_true = y_true + targets_text
            y_pred = y_pred + [tuple(idx2word[i.item()] for i in x[:-1]) for x in ret_object['predicted_indices']]
            
    return y_true, y_pred


def strict(target, prediction):
    return target == prediction


def partial(target, prediction):
    for t, p in zip(target, prediction):
        if p not in t:
            return False

    return True


def evaluate(y_true, y_pred, compare=strict):

    y_pred = copy.copy(y_pred)

    for target, i in zip(y_true, range(len(y_pred))):

        if compare(target, y_pred[i]):
            y_pred[i] = target

    y_true = list(map(lambda x : str(hash(x)), y_true))
    y_pred = list(map(lambda x : str(hash(x)), y_pred))

    return skmetrics.classification_report(y_true, y_pred, digits=3)