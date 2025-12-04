#######################################################
# Implementation of bias computation in contextual embeddings
# Based on the method proposed by Kurita et al. (2019)
# 
# Here there should be the logic for computing bias scores given 
# a set of target and attribute words using contextual embeddings.
#######################################################

import torch
from torch.nn.functional import softmax
from transformers import BertForMaskedLM, BertTokenizer

from .utils.bert_utils import BertForMaskedLMWithDebiasing

from matplotlib import pyplot as plt

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny", do_lower_case=True)
bert_model = BertForMaskedLMWithDebiasing.from_pretrained("prajjwal1/bert-tiny")
bert_model.eval()

def get_mask_index(inputs):
    return (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
        as_tuple=True)[0]

def get_mask_fill_probabilities(sentence, fill_words):
    """
    For a given sentence with one [MASK] token, compute the probabilities
    of filling the mask with each of the fill words.
    """
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # retrieve index of [MASK]
    mask_token_index = get_mask_index(inputs)

    with torch.no_grad():
        logits = bert_model(**inputs).logits

        # get logits for the mask token
        mask_token_logits = logits[0, mask_token_index]

        # compute probabilities
        probs = softmax(mask_token_logits, dim=-1)[0]

        # get probabilities for the fill words
        fill_word_probs = {}
        for word in fill_words:
            token_id = tokenizer.convert_tokens_to_ids(word)
            fill_word_probs[word] = probs[token_id].item()

    return fill_word_probs

def compute_association(p_tgt, p_prior):
    res = torch.log(torch.tensor(p_tgt + 1e-10)) - torch.log(torch.tensor(p_prior + 1e-10))
    return res.item()

def compute_bias_score(
        sentence: str, 
        subject_words: tuple[str, str], 
        attribute_words: list[str]):
    '''
    Compute bias scores.
    
    Args:
        sentence (str): a sentence template with [SUBJ] and [ATTR] placeholders
        subject_words (tuple[str, str]): words representing the subjects 
            (e.g., ("he", "she"))
        attribute_words (list of str): words representing the attributes 
            (e.g., ["programming", "nursing"])

    Returns:
        dict: mapping from attribute word to bias score
    '''
    # sanity checks
    assert "[SUBJ]" in sentence and "[ATTR]" in sentence, \
        "Sentence must contain [SUBJ] and [ATTR] placeholders."
    assert len(subject_words) == 2, \
        "There must be exactly two subject words."
        
    # 1. sentence has to include "[SUBJ]", "[ATTR]"

    # 3. Replace [SUBJ] with [MASK] to compute target probabilities
    prior_subj_probs = get_mask_fill_probabilities(
        sentence.replace("[ATTR]", "[MASK]").replace("[SUBJ]", "[MASK]"), 
        subject_words)

    # print("Prior subject probabilities:", prior_subj_probs)
    # print("Log Prior probabilities:", torch.log(torch.tensor(prior_subj_probs["he"]/prior_subj_probs["she"])))
    
    scores = {}
    for attr in attribute_words:
        # 2. Replace [SUBJ] with [MASK] to compute target probabilities
        target_sentence = sentence.replace("[SUBJ]", "[MASK]").replace("[ATTR]", attr)
        target_probs = get_mask_fill_probabilities(target_sentence, subject_words)
        
        # print(f"Attribute: {attr}")
        associations = {}
        mw, fw = subject_words
        p_tgt_0 = target_probs[mw]
        p_tgt_1 = target_probs[fw]
        p_prior_0 = prior_subj_probs[mw]
        p_prior_1 = prior_subj_probs[fw]
        # 4. Get the association for each subject word
        associations[mw] = compute_association(p_tgt_0, p_prior_0)
        associations[fw] = compute_association(p_tgt_1, p_prior_1)

        # 4b. Get the association difference (the bias score)
        scores[attr] = associations[subject_words[0]] - associations[subject_words[1]]

    return scores
    
def plot_bias_scores(scores, save_path=None):
    # plot bar chart of scores
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(scores)), [scores[attr].item() for attr in scores], tick_label=list(scores.keys()))
    plt.xticks(rotation=90)
    plt.ylabel("Bias Score")
    plt.title("Bias Scores for In-Demand Tech Skills")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_bias_pie_chart(scores, save_path=None):
    # plot pie chart of positive vs negative scores
    positive_scores = sum(1 for score in scores.values() if score.item() > 0)
    negative_scores = sum(1 for score in scores.values() if score.item() < 0)

    plt.figure()
    plt.pie([positive_scores, negative_scores], labels=["Positive", "Negative"], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    plt.title("Distribution of Bias Scores for In-Demand Tech Skills")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()