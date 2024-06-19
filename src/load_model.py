# src/load_model.py
"""
Script to load the trained BERT4Rec model.
"""
import os
import torch
from transformers import BertTokenizer
import pickle

from model import BERT4RecWithFeatures

def load_model(model_path):
    """
    Load the trained BERT4Rec model and its components.

    Parameters:
    - model_path (str): Path where the model and components are saved.

    Returns:
    - model (nn.Module): Loaded BERT4Rec model.
    - product_encoder (LabelEncoder): Encoder for product IDs.
    - tokenizer (BertTokenizer): Tokenizer for encoding sequences.
    """
    # Load model
    num_product_features = 6  # Adjust if needed based on your model
    num_labels = len(product_encoder.classes_)  # Load from the saved encoder
    model = BERT4RecWithFeatures(num_product_features, num_labels)
    model_save_path = os.path.join(model_path, 'saved_models/bert4rec_model/pytorch_model.bin')
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # Load product encoder
    encoder_save_path = os.path.join(model_path, 'saved_models/bert4rec_model/product_encoder.pkl')
    with open(encoder_save_path, 'rb') as f:
        product_encoder = pickle.load(f)
    
    # Load tokenizer
    tokenizer_save_path = os.path.join(model_path, 'saved_models/bert4rec_model/tokenizer')
    tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)
    
    return model, product_encoder, tokenizer

if __name__ == "__main__":
    model_path = '/saved_models/bert4rec_model'
    model, product_encoder, tokenizer = load_model(model_path)
