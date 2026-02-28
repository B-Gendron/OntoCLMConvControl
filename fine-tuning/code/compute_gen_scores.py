import os
import json
import argparse
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score # RoBERTa large model is used

def load_texts(file_path):
    '''
        Loads lines from a text file, ensuring it has 3200 (= 32 samples * 100 steps) lines.
    '''
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    if len(lines) != 3200:
        print(f"Warning: {file_path} has {len(lines)} lines instead of 3200.")
    return lines

def compute_bleu(predictions, references):
    '''
        Computes BLEU score using sacrebleu.
    '''
    return sacrebleu.corpus_bleu(predictions, [references]).score

def compute_rouge(predictions, references):
    '''
        Computes ROUGE scores using rouge_score.
    '''
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(pred, ref) for pred, ref in zip(predictions, references)]
    
    # Average scores
    rouge1 = sum(s["rouge1"].fmeasure for s in scores) / len(scores)
    rouge2 = sum(s["rouge2"].fmeasure for s in scores) / len(scores)
    rougeL = sum(s["rougeL"].fmeasure for s in scores) / len(scores)
    
    return {"ROUGE-1": rouge1, "ROUGE-2": rouge2, "ROUGE-L": rougeL}

def compute_bert_score(predictions, references):
    '''
        Computes BERTScore using bert-score.
    '''
    P, R, F1 = score(predictions, references, lang="en", rescale_with_baseline=True)
    return {"BERT-P": P.mean().item(), "BERT-R": R.mean().item(), "BERT-F1": F1.mean().item()}

def evaluate_scores(subfolder):
    '''
        Evaluates BLEU, ROUGE, and BERTScore for the given subfolder.
    '''
    base_path = os.path.join("..", "gen_quality", subfolder)
    
    reference_file = os.path.join(base_path, f"{subfolder}_raw.txt")
    predictions_file = os.path.join(base_path, f"{subfolder}_raw_shuffled.txt")

    if not os.path.exists(reference_file) or not os.path.exists(predictions_file):
        print(f"Error: One or both reference/predictions files are missing in subfolder {subfolder}.")
        return
    
    # load generated contents
    references = load_texts(reference_file)
    predictions = load_texts(predictions_file)

    # compute generation quality metrics
    bleu_score = compute_bleu(predictions, references)
    rouge_scores = compute_rouge(predictions, references)
    bert_scores = compute_bert_score(predictions, references)

    print(bleu_score, rouge_scores, bert_scores)

    if references is None or predictions is None:
        return None

    return {
        "BLEU": bleu_score,
        "ROUGE": rouge_scores,
        "BERT_SCORE": bert_scores,
    }

if __name__ == "__main__":
    results = {}
    base_dir = os.path.join("..", "gen_quality")

    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        # check if the folder is not empty
        if os.path.isdir(subfolder_path) and os.listdir(subfolder_path):
            scores = evaluate_scores(subfolder)
            if scores:
                results[subfolder] = scores

    with open("../results/generation_quality_ref_ref.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
