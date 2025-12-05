from torch.utils.data import Dataset
import pandas as pd

PRONOUNS_DATA_PATH = "data/pronouns_gender_bias.csv"
OCCUPATION_DATA_PATH = "data/occupation_data.csv"

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
    # Load words and sentences from data source
    pass

class BiasDataset(Dataset):
    
    def __init__(self, tokenizer):

        self.words = words
        self.sentences = sentences
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
        # input_ids, words, pronouns