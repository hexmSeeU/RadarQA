from bert_score import BERTScorer
import os
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from evaluate import load
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from tqdm import tqdm



bert_model_path = "/path/to/bert-base-uncased"
nltk_data_path = "/path/to/nltk_data"

nltk.data.path.append(nltk_data_path)

# 创建BERTScorer
bert_scorer = BERTScorer(model_path=bert_model_path, model_type="bert-base-uncased")

# 创建ROUGEScorer
rouge_l_scorer = rouge_scorer.RougeScorer(['rougeLsum'])

# 创建meteor_metric
meteor_metric = load("meteor")

FOLDER = "/path/to/model_result_folder" 

if __name__ == "__main__":
    bert_sum_score = 0
    bleu_sum_score = 0
    meteor_sum_score = 0
    rouge_sum_score = 0
    for filename in tqdm(os.listdir(FOLDER)):
        file_path = os.path.join(FOLDER, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
        response_value = data["response"]
        label = data["labels"]
        
        # BertScore
        P, R, F1 = bert_scorer.score([label], [response_value])
        
        # Bleu
        response_tokens = word_tokenize(response_value)
        label_tokens = word_tokenize(label)
        
        bleu_score = sentence_bleu([label_tokens], response_tokens)
        
        # Meteor
        meteor_score = meteor_metric.compute(predictions=[response_value], references=[label])
                
        # Rouge_L
        rouge_scores = rouge_l_scorer.score(target=label, prediction=response_value)
        
        rouge_score = rouge_scores['rougeLsum'].fmeasure
        
        bert_sum_score += P.mean()
        bleu_sum_score += bleu_score
        meteor_sum_score += meteor_score['meteor']
        rouge_sum_score += rouge_score
    print(f"Average Bert score: {bert_sum_score / len(os.listdir(FOLDER))}")
    print(f"Average BLEU score: {bleu_sum_score / len(os.listdir(FOLDER))}")
    print(f"Average meteor: {meteor_sum_score / len(os.listdir(FOLDER))}")
    print(f"Average rougeLsum: {rouge_sum_score / len(os.listdir(FOLDER))}")
    

                