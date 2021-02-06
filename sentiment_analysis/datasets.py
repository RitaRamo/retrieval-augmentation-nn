from torch.utils.data import Dataset
import faiss
import torch
from variables import *

class SADataset(Dataset):

        def __init__(self, sents_original, sents_ids, lens, labels, bert_model):
            self.sents_original = sents_original
            self.sents_ids = sents_ids
            self.lens = lens
            self.len_dataset = len(sents_ids)
            self.labels = labels
            self.model = bert_model

        def __getitem__(self, i):
            bert_text = self.model.encode(self.sents_original[i][:MAX_LEN])

            return bert_text, torch.tensor(self.sents_ids[i]).long(), torch.tensor(self.lens[i]).long(), torch.tensor(
                self.labels[i], dtype=torch.float64)

        def __len__(self):
            return self.len_dataset

class TrainRetrievalDataset(Dataset):

        def __init__(self, train_sents, bert_model):
            if DEBUG:
                self.train_sents=train_sents[:40] 
            else:
                self.train_sents=train_sents
            self.dataset_size = len(self.train_sents)
            self.model = bert_model

        def __getitem__(self, i):
            bert_text = self.model.encode(self.train_sents[i][:MAX_LEN])
            return bert_text
    
        def __len__(self):
            return self.dataset_size

class TextRetrieval():

    def __init__(self, train_dataloader, labels):
        dim_examples = 768
        self.datastore = faiss.IndexFlatL2(dim_examples)
        
        if MODEL_TYPE== "SAR_bert":
            self.labels = labels
            self.neg_bert_embedding = torch.zeros(dim_examples)
            self.pos_bert_embedding = torch.zeros(dim_examples)
            self.number_of_neg = 0.0
            self.number_of_pos = 0.0
            self._add_examples_SAR_bert(train_dataloader)
            self.neg_bert_embedding = self.neg_bert_embedding/self.number_of_neg
            self.pos_bert_embedding = self.pos_bert_embedding/self.number_of_pos

        else:
            self._add_examples(train_dataloader)

    def _add_examples(self, train_dataloader):

        print("\nAdding input examples to datastore (retrieval)")
        for i, (text_bert) in enumerate(train_dataloader):
            self.datastore.add(text_bert.cpu().numpy())
            if i%5==0:
                print("batch, n of examples", i, self.datastore.ntotal)
        print("finish retrieval")

    def _add_examples_SAR_bert(self, train_dataloader):

        print("\nAdding input examples to datastore (retrieval)")

        for i, (text_bert) in enumerate(train_dataloader):
            batch_size = text_bert.size()[0]
            
            neg_sentences = text_bert[torch.tensor(self.labels[i*batch_size:i*batch_size+batch_size])==0]
            self.number_of_neg +=neg_sentences.size()[0]
            self.neg_bert_embedding += torch.sum(neg_sentences, dim=0)
           
            pos_sentences = text_bert[torch.tensor(self.labels[i*batch_size:i*batch_size+batch_size])==1]
            self.number_of_pos += pos_sentences.size()[0]
            self.pos_bert_embedding += torch.sum(pos_sentences, dim=0)

            self.datastore.add(text_bert.cpu().numpy())
        print("finish retrieval")

    def retrieve_nearest_for_train_query(self, query_text, k=2):

        D, I = self.datastore.search(query_text, k)
        nearest_input = I[:,1]

        return nearest_input

    def retrieve_nearest_for_val_or_test_query(self, query_text, k=1):

        D, I = self.datastore.search(query_text, k)
        nearest_input = I[:,0]

        return nearest_input
