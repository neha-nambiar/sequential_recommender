# model.py
"""
Defines the BERT4Rec model with product feature embeddings.
"""

import torch
from torch import nn
from transformers import BertModel

class BERT4RecWithFeatures(nn.Module):
    def __init__(self, num_product_features, num_labels, bert_model_name='bert-base-uncased'):
        super(BERT4RecWithFeatures, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.product_embedding = nn.Linear(num_product_features, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, product_features):
        outputs = self.bert(input_ids)
        sequence_output = outputs.last_hidden_state
        
        # Combine with product features
        product_embedding_output = self.product_embedding(product_features)
        combined_output = sequence_output + product_embedding_output.unsqueeze(1)  # Adjust shapes if needed
        
        logits = self.classifier(combined_output[:, -1, :])  # Use last token output for classification
        return logits
