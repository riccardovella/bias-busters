#######################################################
# Implementation of bias removal from contextual embeddings.
# This is performed by optimizing a linear transformation that, when applied 
# to the embeddings, minimizes bias while preserving semantic information.
# 
# Here there should be the logic for computing the linear transformation
# and applying it to the embeddings.
#######################################################

import pandas as pd
import numpy as np
import torch
from utils.bert_utils import Debiaser, BertForMaskedLMWithDebiasing
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from bias_computation import compute_prior_probabilities, compute_bias_scores_from_logits, compute_association

from utils.data_utils import load_pronouns, BiasDataset

torch.manual_seed(42)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, words, sentence_templates, tokenizer):
        self.words = words
        self.sentence_templates = sentence_templates
        self.tokenizer = tokenizer

    def __len__(self):
        return self.words.shape[0]

    def __getitem__(self, idx):
        # Fill in the sentence template with the word
        self.sentences = [
                template.replace("[ATTR]", word) 
            for template, word in zip(self.sentence_templates, self.words)
        ]
        ret = {}
        ret['input_ids'] = self.tokenizer(
            self.sentences[idx], 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=16)
        ret['words'] = self.words[idx]
        return ret

def train_debiaser(dataloader, num_epochs=10, learning_rate=1e-3, weight_decay=1e-5, alpha=1.0):
    # Optimization to find a linear transformation
    # that minimizes bias in the embeddings.

    debiaser = Debiaser()
    # debiaser = torch.nn.Identity()

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")

    model_masked_lm = BertForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
    bert_model = model_masked_lm.bert
    mlm_head = model_masked_lm.cls
    bert_model.eval()  # freeze BERT parameters
    mlm_head.eval()  # freeze MLM head parameters
    debiaser.eval() 

    optimizer = torch.optim.Adam(
        debiaser.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = Loss(alpha=alpha)

    # Compute prior probabilities for subject words 
    sentence_template_1 = "[SUBJ] is a [ATTR]"
    sentence_template_2 = "[SUBJ] can [ATTR]"
    sentence_template_3 = "[SUBJ] is [ATTR]"

    subj_words = load_pronouns()


    # Create a list of sentence templates
    sentence_templates = [sentence_template_1, sentence_template_2, sentence_template_3]

    # Create a dictionary to store the prior probabilities
    prior_probs = {}

    # For each sentence template, compute the prior probabilities for each subject word
    for sentence_template in sentence_templates:
        inputs = tokenizer(
            sentence_template.replace("[SUBJ]", "[MASK]").replace("[ATTR]", "[MASK]"), 
            return_tensors='pt', padding=True, truncation=True, max_length=16)
        outputs = bert_model(**inputs)
        logits = mlm_head(outputs[0])
        for subj_word in subj_words:
            p = compute_prior_probabilities(inputs, logits, subj_word)
            for s, p in p.items():
                prior_probs[(s, sentence_template)] = p
        

    print(prior_probs)
    
    def print_most_likely_words(with_debiaser=True):
        debiaser.eval() 
        # Prepare sentences for target probability computation
        #attr_words = ["engineer", "nurse", "programmer", "teacher", "prostitute", "mother", "father"]
        sub_words = ["he", "she", "Mary", "John"]
        sentence_template = "[SUBJ] is a [ATTR]"
        sentences_filled = [
            sentence_template.replace("[SUBJ]", subj).replace("[ATTR]", "[MASK]")
            for subj in sub_words
        ]

        inputs = tokenizer(
            sentences_filled, 
            return_tensors='pt', padding=True, truncation=True, max_length=16)
        mask_idxs = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        outputs = bert_model(**inputs)
        # take the embeddings corresponding to the [MASK] tokens for each sentence
        last_hidden_state = outputs.last_hidden_state
        embeddings = last_hidden_state[torch.arange(len(mask_idxs)), mask_idxs, :]

        # SHOULD WE DEBIAS JUST THE [SUBJ] TOKENS OR ALL THE TOKENS?
        if with_debiaser:
            debiased_embeddings = debiaser(embeddings)
        else:
            debiased_embeddings = embeddings

        last_hidden_state[torch.arange(len(mask_idxs)), mask_idxs, :] = debiased_embeddings

        logits = mlm_head(last_hidden_state)

        # get logits for the mask token
        mask_token_logits = logits[torch.arange(len(mask_idxs)), mask_idxs]

        # compute probabilities
        probs = torch.nn.functional.softmax(mask_token_logits, dim=-1)

        # get top 5 probabilities for each sentence
        top_k = 5
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        for i, subj in enumerate(sub_words):
            print(f"Attribute: {subj}")
            for j in range(top_k):
                token_id = top_indices[i, j].item()
                token = tokenizer.convert_ids_to_tokens(token_id)
                prob = top_probs[i, j].item()
                print(f"  {token}: {prob:.4f}")
            print()

    print_most_likely_words(with_debiaser=False)

    # Training loop for debiaser
    for epoch in range(num_epochs):  # number of epochs
        debiaser.train()
        total_loss = 0
        for data in dataloader:
            optimizer.zero_grad()
            
            inputs = data['inputs']
            sentence_templates = data['template']

            # Prepare sentences for target probability computation
            mask_idxs = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

            outputs = bert_model(**inputs)
            # take the embeddings corresponding to the [MASK] tokens for each sentence
            last_hidden_state = outputs.last_hidden_state
            mask_embeddings = last_hidden_state[torch.arange(len(mask_idxs)), mask_idxs, :]
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings

            # SHOULD WE DEBIAS JUST THE [SUBJ] TOKENS OR ALL THE TOKENS?
            debiased_embeddings = debiaser(mask_embeddings)
            with torch.no_grad():
                debiased_embeddings_cls = debiaser(cls_embeddings)

            last_hidden_state[torch.arange(len(mask_idxs)), mask_idxs, :] = debiased_embeddings

            logits = mlm_head(last_hidden_state)

            # get logits for the mask token
            mask_token_logits = logits[torch.arange(len(mask_idxs)), mask_idxs]

            # compute probabilities
            probs = torch.nn.functional.softmax(mask_token_logits, dim=-1)

            # get probabilities for the fill words
            
            bias_score = 0
            for word_f, word_m in subj_words:
                token_id_m = tokenizer.convert_tokens_to_ids(word_m)
                token_id_f = tokenizer.convert_tokens_to_ids(word_f)
                target_probs_m = probs[:, token_id_m]
                target_probs_f = probs[:, token_id_f]

                # compute bias score
                ass_f = lambda t_p, p_p: torch.log(t_p + 1e-10) - torch.log(p_p + 1e-10)

                association_scores_m = ass_f(target_probs_m, torch.tensor(prior_probs[word_m, sentence_template]))
                association_scores_f = ass_f(target_probs_f, torch.tensor(prior_probs[word_f, sentence_template]))
                bias_score += association_scores_m - association_scores_f

            bias_score /= len(subj_words)

            # get loss
            loss = criterion(bias_score, cls_embeddings, debiased_embeddings_cls)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(total_loss)

    print_most_likely_words(with_debiaser=True)

class Loss(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(Loss, self).__init__()
        self.alpha = alpha

    def forward(self, bias_scores, e_0, e_1):
        # Compute loss as sum of squared bias scores plus the cosine distance
        # between the original embeddings e_0 and debiased embeddings e_1
        loss_bias = bias_scores.pow(2).sum() 
        loss_preservation = 1 - torch.nn.functional.cosine_similarity(e_0, e_1).mean()
        loss = self.alpha * loss_bias + (1 - self.alpha) * loss_preservation
        return loss

def load_data(tokenizer):
    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")

    # Load all data
    data_attr = pd.read_csv('data/atributes/attributes_training_set_(X_is_Y).csv')
    data_occs = pd.read_csv('data/occupations/occupations_training_set.csv', header=None, names=['OCCUPATION'])
    data_skills = pd.read_csv('data/skills/skills_training_set.csv')
    data_occs2 = pd.read_csv('data/occupations/job_nodup.csv')

    words_a = data_attr['ATTRIBUTE'].to_numpy()
    # labels = data_attr['VALENCE'].to_numpy()
    sentence_template = "[SUBJ] is [ATTR]"
    sentences_a = [
        sentence_template.replace("[ATTR]", word) 
        for word in words_a
    ]

    words_o = data_occs['OCCUPATION'].to_numpy()
    sentence_template = "[SUBJ] is a [ATTR]"
    sentences_o = [
        sentence_template.replace("[ATTR]", word) 
        for word in words_o
    ]

    words_o2 = pd.concat([data_occs2['F_S'], data_occs2['M_S'], data_occs2['N']]).to_numpy()
    # remove nan values
    words_o2 = words_o2[~pd.isna(words_o2)]
    sentences_o2 = [
        sentence_template.replace("[ATTR]", word) 
        for word in words_o2
    ]

    words_s = pd.concat([data_skills['Males'], data_skills['Females']]).to_numpy()
    sentence_template = "[SUBJ] can [ATTR]"
    sentences_s = [
        sentence_template.replace("[ATTR]", word) 
        for word in words_s
    ]

    words = np.concatenate([words_a, words_o, words_o2, words_s])
    sentences = np.concatenate([sentences_a, sentences_o, sentences_o2, sentences_s])

    dataset = Dataset(words, 
                      sentences, 
                      tokenizer)
    return dataset

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
dataset = BiasDataset(tokenizer)
print(dataset.__len__(), "samples loaded.")
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True)

num_epochs = 10
learning_rate = 1e-3
weight_decay = 1e-5
alpha = 0.5
train_debiaser(dataloader, num_epochs, learning_rate, weight_decay, alpha)