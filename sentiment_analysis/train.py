from models import SARModel
from datasets import SADataset, TrainRetrievalDataset, TextRetrieval
import torch.nn as nn
import time
import torch.optim as optim
from collections import OrderedDict, Counter
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from utils import *


torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    train_sents, train_labels, test_sents, test_labels = import_data_from_files()

    train_sents, val_sents, train_labels, val_labels = train_test_split(train_sents,train_labels, test_size=0.1, random_state=42)

    train_words = " ".join(train_sents).split()
    words_counter = Counter(train_words)
    words = [w for w in words_counter.keys() if words_counter[w] > MIN_FREQ_WORD]
    vocab = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN] + words

    token_to_id = OrderedDict([(value, index)
                               for index, value in enumerate(vocab)])

    model = SARModel(len(vocab),
                     token_to_id[PAD_TOKEN],
                     token_to_id,
                     device)

    sents_with_tokens = [text.split() for text in train_sents]
    train_sents_ids, train_lens = convert_sent_tokens_to_ids(sents_with_tokens, MAX_LEN, token_to_id)

    val_sents_with_tokens = [text.split() for text in val_sents]
    val_sents_ids, val_lens = convert_sent_tokens_to_ids(val_sents_with_tokens, MAX_LEN, token_to_id)

    test_sents_with_tokens = [text.split() for text in test_sents]
    test_sents_ids, test_lens = convert_sent_tokens_to_ids(test_sents_with_tokens, MAX_LEN, token_to_id)

    bert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    train_retrieval_iterator = torch.utils.data.DataLoader(
        TrainRetrievalDataset(train_sents,bert_model),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    text_retrieval=TextRetrieval(train_retrieval_iterator, train_labels)
    target_lookup = torch.tensor(train_labels)

    print("Loading the two representations for the binary targets")
    if MODEL_TYPE == "BASELINE":
        target_representations = torch.zeros(2, HIDDEN_DIM).to(device)

    elif MODEL_TYPE== "SAR_-11":
        target_representations = torch.ones(2, HIDDEN_DIM).to(device)
        target_representations[0, :] = -1.0

    elif MODEL_TYPE == "SAR_avg":
        train_neg_sents_ids = torch.tensor(train_sents_ids)[torch.tensor(train_labels) == 0]
        train_pos_sents_ids = torch.tensor(train_sents_ids)[torch.tensor(train_labels) == 1]

        negs_embeddings = model.embedding(train_neg_sents_ids.long())
        pos_embeddings = model.embedding(train_pos_sents_ids.long())

        avg_negs_embedding = negs_embeddings.mean(1).mean(0)
        avg_pos_embedding = pos_embeddings.mean(1).mean(0)

        target_representations = torch.cat((avg_negs_embedding.unsqueeze(0), avg_pos_embedding.unsqueeze(0)), dim=0)

    elif MODEL_TYPE == "SAR_norm":
        train_neg_sents_ids=torch.tensor(train_sents_ids)[torch.tensor(train_labels)==0]
        train_pos_sents_ids=torch.tensor(train_sents_ids)[torch.tensor(train_labels)==1]
       
        negs_embeddings=model.embedding(train_neg_sents_ids.long())
        pos_embeddings=model.embedding(train_pos_sents_ids.long())

        negs_norms=negs_embeddings.norm(p=2, dim=-1)  
        all_sentences_weighted_negs_embedding = torch.sum(
            negs_embeddings * negs_norms.unsqueeze(-1), dim=1) / torch.sum(negs_norms, dim=-1).unsqueeze(-1)
        weighted_negs_embedding = all_sentences_weighted_negs_embedding.mean(0)
 
        pos_norms=pos_embeddings.norm(p=2, dim=-1)  
        all_sentences_weighted_pos_embedding = torch.sum(
            pos_embeddings * pos_norms.unsqueeze(-1), dim=1) / torch.sum(pos_norms, dim=-1).unsqueeze(-1)
        weighted_pos_embedding = all_sentences_weighted_pos_embedding.mean(0)
 
        target_representations= torch.cat((weighted_negs_embedding.unsqueeze(0), weighted_pos_embedding.unsqueeze(0)), dim=0)
   
    elif MODEL_TYPE =="SAR_bert":
        target_representations= torch.cat((text_retrieval.neg_bert_embedding.unsqueeze(0), text_retrieval.pos_bert_embedding.unsqueeze(0)), dim=0)

    else:
        raise Exception("Unknown model")

    train_iterator = torch.utils.data.DataLoader(
        SADataset(train_sents, train_sents_ids, train_lens, train_labels, bert_model),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_iterator = torch.utils.data.DataLoader(
        SADataset(val_sents, val_sents_ids, val_lens, val_labels, bert_model),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_iterator = torch.utils.data.DataLoader(
        SADataset(test_sents, test_sents_ids, test_lens, test_labels, bert_model),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    def train(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        all_preds=[]
        all_labels = []

        model.train()

        for batch, (text_bert, text, text_lengths, label) in enumerate(iterator):
            optimizer.zero_grad()

            retrieved_neighbors_index = text_retrieval.retrieve_nearest_for_train_query(text_bert.numpy()) 
            target_neighbors=target_lookup[retrieved_neighbors_index]
            target_neighbors_representations = target_representations[target_neighbors]

            text_lengths, sort_ind = text_lengths.sort(dim=0, descending=True)
            text_lengths = text_lengths.to(device)
            text = text[sort_ind].to(device)
            label = label[sort_ind].to(device)       
            target_neighbors_representations = target_neighbors_representations[sort_ind].to(device) 
            text= text.permute(1, 0)

            predictions = model(text, text_lengths, target_neighbors_representations).squeeze(1)
            
            all_preds.append(predictions)
            all_labels.append(label)

            loss = criterion(predictions, label)

            acc = binary_accuracy(predictions, label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if batch %20==0:
                print(f'\tTrain Loss: {(epoch_loss/ (batch+1)):.4f} | Train Acc: {(epoch_acc/ (batch+1)) * 100:.4f}%')

        all_preds = torch.cat(all_preds, dim=-1)
        all_labels = torch.cat(all_labels, dim=-1)

        f1 = f1score(all_preds, all_labels)
        print(f'\tTrain Loss: {(epoch_loss/ (batch+1)):.4f} | Train Acc: {(epoch_acc/ (batch+1)) * 100:.4f}% | Train f1-score {f1:.4f}')

        return epoch_loss / len(iterator), epoch_acc / len(iterator), f1

    def evaluate(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        model.eval()

        all_preds=[]
        all_labels = []

        with torch.no_grad():
            for batch, (text_bert, text, text_lengths, label) in enumerate(iterator):

                retrieved_neighbors_index = text_retrieval.retrieve_nearest_for_val_or_test_query(text_bert.numpy()) 
                target_neighbors=target_lookup[retrieved_neighbors_index]
                target_neighbors_representations = target_representations[target_neighbors]

                text_lengths, sort_ind = text_lengths.sort(dim=0, descending=True)
                text_lengths = text_lengths.to(device)
                text = text[sort_ind].to(device)
                label = label[sort_ind].to(device)       
                target_neighbors_representations = target_neighbors_representations[sort_ind].to(device) 
                text= text.permute(1, 0)
          
                predictions = model(text, text_lengths, target_neighbors_representations).squeeze(1)

                loss = criterion(predictions, label)

                all_preds.append(predictions)
                all_labels.append(label)

                acc = binary_accuracy(predictions, label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

                if batch %20==0:
                    print(f'\tVal (or Test) Loss: {(epoch_loss/ (batch+1)):.4f} | Val (or Test) Acc: {(epoch_acc/ (batch+1)) * 100:.4f}%')

        all_preds = torch.cat(all_preds, dim=-1)
        all_labels = torch.cat(all_labels, dim=-1)

        f1 = f1score(all_preds, all_labels)

        return epoch_loss / len(iterator), epoch_acc / len(iterator), f1

    best_valid_acc = 0
    counter_without_improvement = 0

    for epoch in range(N_EPOCHS):

        if counter_without_improvement == 12:
            break

        if counter_without_improvement > 0 and counter_without_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        start_time = time.time()

        train_loss, train_acc, train_f1 = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc, valid_f1 = evaluate(model, val_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_acc > best_valid_acc:
            counter_without_improvement = 0
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'SAR_model'+MODEL_TYPE+str(MULTI_ATTENTION)+str(WITHOUT_RETRIEVED_MEMORY)+'.pt')
        else:
            counter_without_improvement += 1

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.4f}% | Train f1-score {train_f1:.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f} |  Val. Acc: {valid_acc * 100:.4f}% | Val. f1-score {valid_f1:.4f}')


    model.load_state_dict(torch.load('SAR_model'+MODEL_TYPE+str(MULTI_ATTENTION)+str(WITHOUT_RETRIEVED_MEMORY)+'.pt'))

    test_loss, test_acc, test_f1 = evaluate(model, test_iterator, criterion)

    print("Model Name:", MODEL_TYPE)
    print("Attention:", MULTI_ATTENTION)
    print("WITHOUT_RETRIEVED_MEMORY:", WITHOUT_RETRIEVED_MEMORY)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.4f}% | Test f1-score {test_f1:.4f} ')
    print("Test entire value: Acc:", test_acc, 'f1-score: ', test_f1)

if __name__ == '__main__':
    main()