import json

from collections import defaultdict
from bert_score import BERTScorer

MODEL = "BASELINE"
MULTILEVEL_ATTENTION =False

if __name__ == "__main__":
    # add bert_scores to coco metrics
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    test_path = "dataset_splits/TEST_COCOTOOLKIT_FORMAT.json"
    if MODEL == "BASELINE":
        generated_sentences_path = "baseline.json"
        scores_path = "baseline_coco_results.json"
    elif MODEL == "SAR_avg":
        generated_sentences_path = "SAR_avg.json"
        scores_path = "SAR_avg_coco_results.json"
    elif MODEL == "SAR_norm":
        if MULTILEVEL_ATTENTION:
            generated_sentences_path = "SAR_norm.json"
            scores_path = "SAR_norm_coco_results.json"
        else:
            generated_sentences_path = "SAR_norm_no_multiattention.json"
            scores_path = "SAR_norm_no_multiattention_coco_results.json"
    elif MODEL == "SAR_bert":
        generated_sentences_path = "SAR_bert.json"
        scores_path = "SAR_bert_coco_results.json"
    elif MODEL == "SAR_norm_wt_m":
        generated_sentences_path = "SAR_norm_wt_m.json"
        scores_path = "SAR_norm_wt_m_coco_results.json"
    else:
        raise Exception("unknown model")

    with open("dataset_splits/TEST_COCOTOOLKIT_FORMAT.json") as json_file:
        test = json.load(json_file)

    dict_imageid_refs = defaultdict(list)
    for ref in test["annotations"]:
        image_id = ref["image_id"]
        caption = ref["caption"]
        dict_imageid_refs[image_id].append(caption)

    # get previous score of coco metrics (bleu,meteor,etc) to append bert_score
    with open(scores_path) as json_file:
        scores = json.load(json_file)

    # get previous generated sentences to calculate bertscore according to refs
    with open(generated_sentences_path) as json_file:
        generated_sentences = json.load(json_file)

    total_precision = 0.0
    total_recall = 0.0
    total_f = 0.0
    for dict_image_and_caption in generated_sentences:
        image_id = dict_image_and_caption["image_id"]
        caption = [dict_image_and_caption["caption"]]
        references = [dict_imageid_refs[image_id]]

        P_mul, R_mul, F_mul = scorer.score(caption, references)
        precision = P_mul[0].item()
        recall = R_mul[0].item()
        f_measure = F_mul[0].item()

        total_precision += precision
        total_recall += recall
        total_f += f_measure

        # calculate bert_score
        key_image_id = str(image_id)
        scores[str(key_image_id)]["BertScore_P"] = precision
        scores[key_image_id]["BertScore_R"] = recall
        scores[key_image_id]["BertScore_F"] = f_measure
        print("\ncaption and score", caption, f_measure)

    n_captions = len(generated_sentences)
    scores["avg_metrics"]["BertScore_P"] = total_precision / n_captions
    scores["avg_metrics"]["BertScore_R"] = total_recall / n_captions
    scores["avg_metrics"]["BertScore_F"] = total_f / n_captions

    # save scores dict to a json
    with open(scores_path, 'w+') as f:
        json.dump(scores, f, indent=2)
