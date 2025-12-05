from torch.utils.data import Dataset
import pandas as pd
import numpy as np

PRONOUNS_DATA_PATH = "data/pronouns_gender_bias.csv"

ATTRIBUTES_DATA_PATH = "data/atributes/attributes_training_set_(X_is_Y).csv"
OCCUPATION_DATA_PATH = "data/occupations/occupations_training_set.csv"
SKILLS_DATA_PATH = "data/skills/skills_training_set.csv"
OCCUPATION_2_DATA_PATH = "data/occupations/job_nodup.csv"

def load_pronouns():
    """Load pronouns from CSV file and return as list of tuples.
    
    Each tuple is in the form (female_pronoun, male_pronoun).
    """
    # Load pronouns from CSV file
    df = pd.read_csv(PRONOUNS_DATA_PATH)
    # make a list of tuples for each row
    pronouns = [(tuple(row)) for row in df.values]
    return pronouns

def _load_all_data():
    # Load all data
    data_attr = pd.read_csv(ATTRIBUTES_DATA_PATH)
    data_occs = pd.read_csv(OCCUPATION_DATA_PATH, header=None, names=['OCCUPATION'])
    data_skills = pd.read_csv(SKILLS_DATA_PATH)
    data_occs2 = pd.read_csv(OCCUPATION_2_DATA_PATH)

    words_a = data_attr['ATTRIBUTE'].to_numpy()
    # labels = data_attr['VALENCE'].to_numpy()
    templates_a = ["[SUBJ] is [ATTR]"] * len(words_a)

    words_o = data_occs['OCCUPATION'].to_numpy()
    templates_o = ["[SUBJ] is a [ATTR]"] * len(words_o)

    words_o2 = pd.concat([data_occs2['F_S'], data_occs2['M_S'], data_occs2['N']]).to_numpy()
    # remove nan values
    words_o2 = words_o2[~pd.isna(words_o2)]
    templates_o2 = ["[SUBJ] is a [ATTR]"] * len(words_o2)

    words_s = pd.concat([data_skills['Males'], data_skills['Females']]).to_numpy()
    templates_s = ["[SUBJ] can [ATTR]"] * len(words_s)

    words = np.concatenate([words_a, words_o, words_o2, words_s])
    templates = np.concatenate([templates_a, templates_o, templates_o2, templates_s])

    return words, templates

class BiasDataset(Dataset):
    def __init__(self, tokenizer):
        words, templates = _load_all_data()

        self.words = words
        self.templates = templates
        self.tokenizer = tokenizer

    def __len__(self):
        return self.words.shape[0]

    def __getitem__(self, idx):
        # construct sentence from template and word
        self.sentences = [
                template.replace("[ATTR]", word) 
            for template, word in zip(self.templates, self.words)]
        # tokenize sentence
        inputs = self.tokenizer(
            self.sentences[idx],    
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=16)
        return {
            'input_ids': inputs['input_ids'],
            'words': self.words[idx],
            'template': self.templates[idx],
        }