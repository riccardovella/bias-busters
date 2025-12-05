from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

import torch
from torch.nn import CrossEntropyLoss

class Debiaser(torch.nn.Module):
    def __init__(self, embedding_dim=128):
        super(Debiaser, self).__init__()
        
        self.linear = torch.nn.Linear(embedding_dim, embedding_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(self.linear.in_features))
            self.linear.bias.fill_(0.0)

    def forward(self, x):
        return self.linear(x)
    
    def save(self, path):
        torch.save({
            'state_dict': self.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])

'''class MaskedLMWithDebiasingFromEmbeddings(torch.nn.Module):
    def __init__(self, config, debiaser=None):
        super().__init__(config)
        self.debiaser = debiaser
        self.cls = BertOnlyMLMHead(config)

    def forward(self, sequence_output, hidden_states, attentions, labels=None):
        if self.debiaser is not None:
            sequence_output = self.debiaser(sequence_output)

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=hidden_states,
            attentions=attentions,
        )
    
    def train(self):
        super().train()
        self.cls.eval()
'''

class BertForMaskedLMWithDebiasing(BertForMaskedLM):
    def __init__(self, config, debiaser=None):
        super().__init__(config)
        self.debiaser = debiaser
        
    '''
    def train(self):
        super().train()
        self.cls.eval()
    '''

    def forward(
        self,
        precomputed_embeddings=None,
        hidden_states=None,
        attentions=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        if precomputed_embeddings is None:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
                **kwargs,
            )

            sequence_output = outputs[0]    # last hidden state from BERT
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states
        else:
            # check that attentions and hidden_states are not None
            if attentions is not None or hidden_states is not None:
                raise ValueError(
                    "When using precomputed embeddings, attentions and hidden_states must be None.")
            sequence_output = precomputed_embeddings

        if self.debiaser is not None:
            sequence_output = self.debiaser(sequence_output)

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=hidden_states,
            attentions=attentions,
        )