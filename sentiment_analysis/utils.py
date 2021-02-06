from variables import *
import json
import numpy as np
from sklearn.metrics import f1_score
import torch

def import_data_from_files():

    with open(DATA_FOLDER+'train_sents.json', 'r') as j:
        train_sents = json.load(j)

    with open(DATA_FOLDER+'train_labels.json', 'r') as j:
        train_labels = json.load(j)

    with open(DATA_FOLDER+'test_sents.json', 'r') as j:
        test_sents = json.load(j)

    with open(DATA_FOLDER+'test_labels.json', 'r') as j:
        test_labels = json.load(j)

    return train_sents, train_labels, test_sents, test_labels

def convert_sent_tokens_to_ids(sent_of_tokens, max_len, token_to_id):

    len_sents = []
    input_sents = np.zeros((len(sent_of_tokens), max_len + 2), dtype=np.int32) + token_to_id[PAD_TOKEN]

    for i in range(len(sent_of_tokens)):
        tokens_to_integer = [token_to_id.get(token, token_to_id[UNK_TOKEN]) for token in sent_of_tokens[i]]
        sent = tokens_to_integer[:max_len]
        sent_with_spetial_tokens = [token_to_id[START_TOKEN]] + sent + [token_to_id[END_TOKEN]]
        input_sents[i, :len(sent_with_spetial_tokens)] = sent_with_spetial_tokens
        len_sents.append(len(sent_with_spetial_tokens))

    return input_sents, len_sents

def adjust_learning_rate(optimizer, shrink_factor):

    print("\nDecaying learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc

def f1score(preds, y):

    predictions = torch.round(torch.sigmoid(preds))

    return f1_score(y.cpu().detach().numpy(), predictions.cpu().detach().numpy(), average="macro")

def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs
