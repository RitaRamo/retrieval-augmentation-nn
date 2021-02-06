import torch
import numpy as np
import torch.nn as nn
from variables import *
import fasttext


class SARModel(nn.Module):

    def __init__(self, vocab_size, pad_idx, token_to_id, device):
        super().__init__()

        self.device=device

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=pad_idx)

        self.lstm = nn.LSTM(EMBEDDING_DIM,
                            HIDDEN_DIM,
                            num_layers=N_LAYERS)

        self.dropout = nn.Dropout(DROPOUT)

        print("Model Name", MODEL_TYPE)

        if MODEL_TYPE == "SAR_-11":
            retrieved_dim = HIDDEN_DIM
        elif MODEL_TYPE == "SAR_avg":
            retrieved_dim = EMBEDDING_DIM  # retrieved target correspond to avg word embeddings from caption
            self.init_c = nn.Linear(retrieved_dim, HIDDEN_DIM)

        elif MODEL_TYPE == "SAR_norm":
            retrieved_dim = EMBEDDING_DIM  # retrieved target correspond to avg embeddings weighted by norm
            self.init_c = nn.Linear(retrieved_dim, HIDDEN_DIM)

        elif MODEL_TYPE == "SAR_bert":
            retrieved_dim = 768  # retrieved target correspond to bert embeddings size
            self.init_c = nn.Linear(retrieved_dim, HIDDEN_DIM)

        if MULTI_ATTENTION:
            print("using our multi attention")
            self.attention = self.attention_multilevel  # proposed attention network
            self.linear_retrieval = nn.Linear(HIDDEN_DIM, retrieved_dim)
            self.hiddens_att = nn.Linear(retrieved_dim, ATTENTION_DIM)  # linear layer to transform hidden states
            self.cat_att = nn.Linear(retrieved_dim, ATTENTION_DIM)
            self.full_multiatt = nn.Linear(ATTENTION_DIM, 1)  # linear layer to calculate values to be softmax-ed
            self.fc = nn.Linear(retrieved_dim, OUTPUT_DIM)

        else:
            print("default attention")
            self.attention = self.attention_baseline
            self.hiddens_att = nn.Linear(HIDDEN_DIM, ATTENTION_DIM)  # linear layer to transform hidden states
            self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

        self.final_hidden_att = nn.Linear(HIDDEN_DIM, ATTENTION_DIM)  # linear layer to transform last hidden state
        self.full_att = nn.Linear(ATTENTION_DIM, 1)  # linear layer to calculate values to be softmax-ed
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

        if DEBUG:
            print("WITHOUT FASTEXT -> DEBUG")
        else:
            print("Loading fasttext embeddings")
            fasttext_embeddings = fasttext.load_model('../image_captioning/embeddings/wiki.en.bin')
            pretrained_embeddings = self._get_fasttext_embeddings_matrix(fasttext_embeddings, vocab_size, EMBEDDING_DIM, token_to_id)
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings))

            self.embedding.weight.requires_grad = False

    def _get_fasttext_embeddings_matrix(self, embeddings, vocab_size, embedding_dim, token_to_id):
        # reduce the matrix of pretrained embeddings according to dataset vocab

        embeddings_matrix = np.zeros(
            (vocab_size, embedding_dim))

        for word, id in token_to_id.items():
            try:
                embeddings_matrix[id] = embeddings.get_word_vector(word)
            except:
                print("How? Fastext has embedding for each token, hence this exception should not happen")
                pass

        return embeddings_matrix

    def attention_multilevel(self, hiddens, final_hidden, retrieved_target):

        # the hidden features receive an affine transformation for this attention, before passing through Eq. 4,
        # to ensure that it has the same dimension of the retrieved target in order to compute Eq. 9 (combine both)
        hiddens = self.linear_retrieval(hiddens)
        hiddens = hiddens.permute(1, 0, 2)
        att1 = self.hiddens_att(hiddens)  # (batch_size, num_hiddens(words), attention_dim)
        att_h = self.final_hidden_att(final_hidden.permute(1, 0, 2))  #(batch_size, 1, attention_dim)
        att = self.full_att(self.tanh(att1 + att_h)).squeeze(2)  #(batch_size, num_hiddens(words), 1)
        alpha = self.softmax(att)  #(batch_size, num_hiddens(words), 1)
        text_context = (hiddens * alpha.unsqueeze(-1)).sum(dim=1)  #(batch_size, hidden_dim)
        text_and_retrieved = torch.cat(([text_context.unsqueeze(1), retrieved_target.unsqueeze(1)]), dim=1)
        att_tr = self.cat_att(text_and_retrieved)
        att_hat = self.full_multiatt(self.tanh(att_tr + att_h)).squeeze(2)  #(batch_size, num_pixels)
        alpha_hat = self.softmax(att_hat)  # (batch_size, num_pixels)
        multilevel_context = (text_and_retrieved * alpha_hat.unsqueeze(2)).sum(dim=1)

        return multilevel_context, alpha_hat

    def attention_baseline(self, hiddens, final_hidden, retrieved_target):

        hiddens = hiddens.permute(1, 0, 2)
        # hiddens torch.Size([64, 611, 512])

        att1 = self.hiddens_att(hiddens)  #(batch_size, num_hiddens(words), attention_dim)
        # att1 torch.Size([64, 611, 512])

        # final_hidden torch.Size([1, 64, 512])
        att2 = self.final_hidden_att(final_hidden.permute(1, 0, 2))  # (batch_size, 1, attention_dim)
        # att2 torch.Size([64, 1, 512])

        att = self.full_att(self.tanh(att1 + att2)).squeeze(2)  # (batch_size, num_hiddens(words), 1)
        # att torch.Size([64, 611])

        alpha = self.softmax(att)  # (batch_size, num_hiddens(words),1)
        # alpha torch.Size([64, 611])

        attention_weighted_encoding = (hiddens * alpha.unsqueeze(-1)).sum(dim=1)  # (batch_size, hidden_dim)
        # attention_weighted_encoding torch.Size([64, 512])

        return attention_weighted_encoding, alpha

    def forward(self, text, text_lengths, target_neighbors_representations):
        # text (sent len, batch size)
        # text torch.Size([144, 64])

        embedded = self.embedding(text)
        # embedded torch.Size([144, 64, 100])
        # embedded (sent len, batch size, emb dim)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to("cpu"))
        # packed_embedded torch.Size([9152, 100])

        init_hidden, init_cell = self.init_hidden_states(target_neighbors_representations)

        packed_output, (hidden, cell) = self.lstm(packed_embedded, (init_hidden, init_cell))
        # packed_output torch.Size([9152, 512])
        # hidden torch.Size([1, 64, 512])
        # cell torch.Size([1, 64, 512])

        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output torch.Size([144, 64, 512])
        # output_lengths torch.Size([64])
        # output (sent len, batch size, hid dim)

        attn_output, alpha = self.attention(output, hidden, target_neighbors_representations)
        # attn_output torch.Size([64, 512])
        # hidden = (batch size, hid dim)
        # self.fc(attn_output) torch.Size([64, 1])

        return self.fc(self.dropout(attn_output))

    def init_hidden_states(self, target_neighbors_representations):

        batch_size = target_neighbors_representations.size()[0]
        init_hidden = torch.zeros(batch_size, HIDDEN_DIM).to(self.device)

        if MODEL_TYPE == "BASELINE":
            init_cell = target_neighbors_representations

        elif MODEL_TYPE == "SAR_-11":
            # already has dimension of the LSTM
            init_cell = target_neighbors_representations

        elif MODEL_TYPE == "SAR_avg":

            if WITHOUT_RETRIEVED_MEMORY:
                # equal to baseline
                init_cell = torch.zeros(BATCH_SIZE, HIDDEN_DIM).to(self.device)
            else:
                init_cell = self.init_c(target_neighbors_representations)


        elif MODEL_TYPE == "SAR_norm":
            init_cell = self.init_c(target_neighbors_representations)

        elif MODEL_TYPE == "SAR_bert":
            init_cell = self.init_c(target_neighbors_representations)

        else:
            raise Exception("that model does not exist")

        return init_hidden.unsqueeze(0), init_cell.unsqueeze(0)