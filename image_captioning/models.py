import torch
from torch import nn
import torchvision
import fasttext
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG=False

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def prepare_encoder_out(self, encoder_out):
        # In pratice uncesICRy: we just use this function to be according to the multi-level attention 
        # In this way, the forward code of both attentions works either
        # we are using the baseline model with this attention or a the ICR model with the other attention
        return encoder_out

    def forward(self, encoder_out, decoder_hidden, additional = None): #additional arg because of multi-level attention
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class MultiLevelAttention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim, retrieved_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(MultiLevelAttention, self).__init__()
        self.linear_retrieval = nn.Linear(encoder_dim, retrieved_dim)

        self.encoder_att = nn.Linear(retrieved_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        
        self.cat_att = nn.Linear(retrieved_dim, attention_dim)  # linear layer to calculate values to be softmax-ed
        self.full_multiatt = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def prepare_encoder_out(self,encoder_out):
        #the image features receive an affine transformation for this attention, before passing through Eq. 4,
        # to ensure that it has the same dimension of the retrieved target in order to compute Eq. 9 (combine both)
        return self.linear_retrieval(encoder_out)  # (batch_size, image_size*image_size, decoder_dim)


    def forward(self, encoder_out, decoder_hidden, retrieved_target):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att_v = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att_h = self.decoder_att(decoder_hidden).unsqueeze(1)  # (batch_size, attention_dim)
        att = self.full_att(self.tanh(att_v + att_h)).squeeze(2)  # Eq.4 (batch_size, num_pixels) 
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        visual_context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        visual_and_retrieved = torch.cat(([visual_context.unsqueeze(1), retrieved_target.unsqueeze(1)]), dim=1)
        att_vr= self.cat_att(visual_and_retrieved) #visual with retrieved target
        att_hat = self.full_multiatt(self.tanh(att_vr + att_h)).squeeze(2)  # (batch_size, num_pixels)
        alpha_hat = self.softmax(att_hat)  # (batch_size, num_pixels)
        multilevel_context=(visual_and_retrieved * alpha_hat.unsqueeze(2)).sum(dim=1)

        return multilevel_context, alpha_hat


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, model_type, multi_attention, attention_dim, embed_dim, decoder_dim, vocab_size, token_to_id, lookup_table, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        
        print("Model", model_type)

        self.model_type = model_type
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.token_to_id = token_to_id
        self.dropout = dropout

        if model_type == "ICR_avg":
            retrieved_dim= self.embed_dim # retrieved target correspond to avg word embeddings from caption
        elif model_type == "ICR_norm":
            retrieved_dim= self.embed_dim # retrieved target correspond to avg embeddings weighted by norm
        elif model_type == "ICR_bert":
            retrieved_dim= 768 # retrieved target correspond to bert embeddings size
        elif model_type == "ICR_norm_wt_m":
            retrieved_dim= self.embed_dim # retrieved target correspond to avg embeddings weighted by norm

        #chamas a attention dependo do modelo...dar erro baseline com attentin nearest
        if multi_attention:
            print("using our multi attention")
            self.attention = MultiLevelAttention(encoder_dim, decoder_dim, attention_dim, retrieved_dim)  # proposed attention network
            self.decode_step = nn.LSTMCell(embed_dim + retrieved_dim, decoder_dim, bias=True)  # decoding LSTMCell

        else: #baseline attention
            print("default attention")
            self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
            self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell

        if model_type== "BASELINE":
            self.init_c = nn.Linear(encoder_dim, decoder_dim) # linear layer to find initial hidden state of LSTMCell
        elif model_type== "ICR_norm_wt_m":
            self.init_c = nn.Linear(encoder_dim, decoder_dim)
        else:
            self.init_c = nn.Linear(retrieved_dim, decoder_dim) 

        #self.init_c = nn.Linear(embed_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        #self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        #self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        
        self.target_lookup= lookup_table #target lookup table of the nearest input examples
        #print("self.lookup_table just to check", self.target_lookup[:10])

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

        if DEBUG:
            print("WITHOUT FASTEXT -> DEBUG")
        else:
            print("pretraining fastext")
            #init embedding layer
            fasttext_embeddings = fasttext.load_model('embeddings/wiki.en.bin')
            pretrained_embeddings = self._get_fasttext_embeddings_matrix(fasttext_embeddings)
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings))

            # pretrained embedings are not trainable by default
            self.embedding.weight.requires_grad = False

    
    def _get_fasttext_embeddings_matrix(self,embeddings):
    # reduce the matrix of pretrained:embeddings according to dataset vocab
        print("loading fasttext embeddings")

        embeddings_matrix = np.zeros(
            (self.vocab_size, self.embed_dim))

        for word, id in self.token_to_id.items():
            try:
                embeddings_matrix[id] = embeddings.get_word_vector(word)
            except:
                print("How? Fastext has embedding for each token, hence this exception should not happen")
                pass

        return embeddings_matrix

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out, retrieved_neighbors_index):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)

        if self.model_type == "BASELINE":
            c = self.init_c(mean_encoder_out)
            target_neighbors_representation= c # the baseline does not have retrieved target
            #it is just for code coherence in respect to ICR Model that needs that extra output

        elif self.model_type == "ICR_avg":
            #print("self target look", self.target_lookup)
            #print("retrieved_neighbors_index",retrieved_neighbors_index)
            target_neighbors=self.target_lookup[retrieved_neighbors_index*5] # each image has 5 captions
            target_neighbors_representation = self.embedding(target_neighbors).mean(1)
            
            # print("target caps", target_neighbors)
            # print("embed of near", target_neighbors_representation.size())
            c = self.init_c(target_neighbors_representation)

        elif self.model_type == "ICR_norm":
            target_neighbors=self.target_lookup[retrieved_neighbors_index*5] # each image has 5 captions
            
            caps_embeddings = self.embedding(target_neighbors)
            caps_norms=caps_embeddings.norm(p=2, dim=-1)
            weighted_embedding = torch.sum(
                caps_embeddings * caps_norms.unsqueeze(-1), dim=1) / torch.sum(caps_norms, dim=-1).unsqueeze(-1)

            target_neighbors_representation = weighted_embedding
            
            c = self.init_c(target_neighbors_representation)

        elif self.model_type == "ICR_bert":
            #this lookup only contains firt caption (hence no need to multiply *5)
            #the lookup already gives the target representation (for efficency we compute bert before)
            target_neighbors_representation=self.target_lookup[retrieved_neighbors_index] 
            # print("embed of near", target_neighbors_representation.size())
            c = self.init_c(target_neighbors_representation)

        elif self.model_type == "ICR_norm_wt_m":
            #equal to ICR norm but without the memory celll being inicialzated with retrieved
            target_neighbors=self.target_lookup[retrieved_neighbors_index*5] # each image has 5 captions
            
            caps_embeddings = self.embedding(target_neighbors)
            caps_norms=caps_embeddings.norm(p=2, dim=-1)
            weighted_embedding = torch.sum(
                caps_embeddings * caps_norms.unsqueeze(-1), dim=1) / torch.sum(caps_norms, dim=-1).unsqueeze(-1)

            target_neighbors_representation = weighted_embedding
            
            c = self.init_c(mean_encoder_out)

        else:
            raise Exception ("no mode model")
        
        return h, c, target_neighbors_representation

    def forward(self, encoder_out, retrieved_neighbors_index, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param retrieved_neighbors_index: for each encoded image, the corresponding index of its nearest neighbour
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        retrieved_neighbors_index = retrieved_neighbors_index[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c,retrieved_target = self.init_hidden_state(encoder_out, retrieved_neighbors_index)  # (batch_size, decoder_dim)
        
        #the image features receive an affine transformation for the multi-leval attention
        #print("encoder out before", encoder_out.size())
        encoder_out= self.attention.prepare_encoder_out(encoder_out) 
        #print("encoder out after", encoder_out.size())

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        #alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t],
                                                                retrieved_target[:batch_size_t]
                                                                )
            #gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            #attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            #alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, sort_ind