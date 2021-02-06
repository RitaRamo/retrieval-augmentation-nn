import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

MODEL_TYPE = "BASELINE"
MULTILEVEL_ATTENTION =False
#BASELINE
#ICR_avg
#ICR_norm
#ICR_bert

#if MODEL_TYPE == "BASELINE":
checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq_'+MODEL_TYPE+str(MULTILEVEL_ATTENTION)+'.pth.tar'
#checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq_'+MODEL_TYPE+'.pth.tar'

print("checkpoint", checkpoint)
# elif MODEL_TYPE == "ICR_avg":
#     checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq_ICR_avg.pth.tar'
# elif MODEL_TYPE == "ICR_norm":
#     checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq_ICR_norm.pth.tar'

# Parameters
# folder with data files saved by create_input_files.py
data_folder = 'dataset_splits'
# base name shared by data files
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'
# model checkpoint
#checkpoint = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'
# word map, ensure it's the same the data was encoded with and the model was trained with
word_map_file = 'dataset_splits/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'
# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """

    train_retrieval_loader = torch.utils.data.DataLoader(
        TrainRetrievalDataset(data_folder, data_name),batch_size=32, shuffle=True, num_workers=1
    )#, pin_memory=True)

    retrieval = ImageRetrieval(decoder.encoder_dim, encoder, train_retrieval_loader, device)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST'),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    hypotheses_bleu = list()
    list_hipotheses = list()

    # For each image
    imgids_so_far = []
    for i, (image, caps, caplens, allcaps, img_id) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out = encoder(image)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        # (1, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        input_imgs = encoder_out.mean(dim=1)
        retrieved_neighbors_index=retrieval.retrieve_nearest_for_val_or_test_query(input_imgs.cpu().numpy())
        #print("encoder_out", encoder_out.size())
        #print("retrieved_neighbour_index", retrieved_neighbors_index.size())
        # We'll treat the problem as having a batch size of k
        # (k, num_pixels, encoder_dim)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
        retrieved_neighbors_index = retrieved_neighbors_index.expand(k)
        # print("encoder_out  after", encoder_out.size())

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor(
            [[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c, retrieved_target = decoder.init_hidden_state(encoder_out, retrieved_neighbors_index)

        encoder_out= decoder.attention.prepare_encoder_out(encoder_out) 

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(
                k_prev_words).squeeze(1)  # (s, embed_dim)

            # (s, encoder_dim), (s, num_pixels)
            awe, _ = decoder.attention(encoder_out, h, retrieved_target)

            # gating scalar, (s, encoder_dim)
            #gate = decoder.sigmoid(decoder.f_beta(h))
            #awe = gate * awe

            h, c = decoder.decode_step(
                torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)

            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(
                    k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                # (s)
                top_k_scores, top_k_words = scores.view(
                    -1).topk(k, 0, True, True)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            
            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(
                set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            incomplete_inds=torch.tensor(incomplete_inds).long()
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds].long()]
            c = c[prev_word_inds[incomplete_inds].long()]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds].long()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 39:
                break
            step += 1

        if k == beam_size:
            complete_seqs.extend(seqs[[incomplete_inds]].tolist())
            complete_seqs_scores.extend(top_k_scores[[incomplete_inds]])

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses for bleu corpus (needs to be a list)
        hypotheses_bleu.append([w for w in seq if w not in {
            word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        # hypotheses for coco (needs to be a string)
        hypotheses = " ".join([rev_word_map[w] for w in seq if w not in {
            word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        # for coco eval
        if img_id.item() not in imgids_so_far:
            list_hipotheses.append({
                "image_id": img_id.item(),
                "caption": hypotheses,
            })
            imgids_so_far.append(img_id.item())

    with open("SAT_discrete.json", 'w+') as f:
        json.dump(list_hipotheses, f, indent=2)

    if MODEL_TYPE == "BASELINE":
        with open("baseline.json", 'w+') as f:
            json.dump(list_hipotheses, f, indent=2)
    elif MODEL_TYPE == "ICR_avg":
        with open("ICR_avg.json", 'w+') as f:
            json.dump(list_hipotheses, f, indent=2)
    elif MODEL_TYPE == "ICR_norm":
        if MULTILEVEL_ATTENTION:
            with open("ICR_norm.json", 'w+') as f:
                json.dump(list_hipotheses, f, indent=2)
        else:
            with open("ICR_norm_no_multiattention.json", 'w+') as f:
                json.dump(list_hipotheses, f, indent=2)
    elif MODEL_TYPE == "ICR_bert":
        with open("ICR_bert.json", 'w+') as f:
            json.dump(list_hipotheses, f, indent=2)
    elif MODEL_TYPE == "ICR_norm_wt_m":
        with open("ICR_norm_wt_m.json", 'w+') as f:
            json.dump(list_hipotheses, f, indent=2)
    else:
        raise Exception("unknow model")

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses_bleu)

    return bleu4


if __name__ == '__main__':
    beam_size = 1
    print("\nBLEU-4 score @ beam size of %d is %.4f." %
          (beam_size, evaluate(beam_size)))
