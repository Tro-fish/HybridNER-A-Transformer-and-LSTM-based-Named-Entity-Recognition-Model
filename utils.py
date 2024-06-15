import utils
import torch
import evaluate

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

CONFIG = {
    'seed' : 12,
    'batch_size' : 64,
    'lr' : 3e-4,
    'weight_decay' : 0.01,
    'hidden_size' : 256,
    'num_heads' : 4,
    'num_encoder_layers' : 6,
    'hidden_dropout_prob': 0.1,
    'use_lstm': True,
    'num_epochs' : 50, 
    'vocab_size' : 30522, 
    'pad_token_id' : 0, 
    'num_labels' : 9,
}

def _expand_mask(mask, tgt_len = None):
    """
        Inputs
            mask.shape = (B, S_L)
        Outputs
            output.shape = (B, 1, T_L, S_L)
    """
    batch_size, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(torch.float)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(torch.float).min)


def compute_metrics(all_predictions, all_labels, label_list):
    seqeval = evaluate.load("seqeval")
    predictions, labels = None, None

    # For evaluation, postprocess for shape.
    predictions = []
    labels = []

    for batch_predictions, batch_labels in zip(all_predictions, all_labels):
        for seq_predictions, seq_labels in zip(batch_predictions, batch_labels):
            predictions.append(seq_predictions)
            labels.append(seq_labels)
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }