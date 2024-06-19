# data_preprocessing.py
"""
Script for preprocessing the session and product data.
"""

import pandas as pd
import numpy as np
import torch 
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

def preprocess_data(sessions_path, products_path, max_seq_length=50):
    """
    Preprocess session and product data.

    Parameters:
    - sessions_path (str): Path to sessions CSV file.
    - products_path (str): Path to products CSV file.
    - max_seq_length (int): Maximum sequence length for BERT tokenizer.

    Returns:
    - X_tensor (torch.Tensor): Encoded session sequences.
    - y_tensor (torch.Tensor): Encoded next item labels.
    - product_feature_tensor (torch.Tensor): Product feature embeddings.
    - product_encoder (LabelEncoder): Encoder for product IDs.
    """

    # Load datasets
    sessions = pd.read_csv(sessions_path)
    products = pd.read_csv(products_path)

    # Encode product IDs
    product_encoder = LabelEncoder()
    sessions['prev_items_encoded'] = sessions['prev_items'].apply(lambda x: product_encoder.fit_transform(eval(x)))
    sessions['next_item_encoded'] = product_encoder.transform(sessions['next_item'])

    # Tokenize sequences
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sessions['encoded_seq'] = sessions['prev_items_encoded'].apply(lambda x: tokenizer.encode(
        x, truncation=True, padding='max_length', max_length=max_seq_length))

    # Prepare tensors
    X = np.vstack(sessions['encoded_seq'].values)
    y = sessions['next_item_encoded'].values
    X_tensor = torch.tensor(X, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Process product features
    products = preprocess_product_features(products, product_encoder)
    product_feature_tensor = torch.tensor(products, dtype=torch.float)

    return X_tensor, y_tensor, product_feature_tensor, product_encoder

def preprocess_product_features(products, product_encoder):
    """
    Encode and normalize product features.

    Parameters:
    - products (DataFrame): DataFrame containing product information.
    - product_encoder (LabelEncoder): Encoder for product IDs.

    Returns:
    - product_embeddings (ndarray): Normalized product features.
    """

    product_features = ['brand', 'color', 'size', 'model', 'material']
    encoders = {}

    for feature in product_features:
        encoder = LabelEncoder()
        products[feature] = products[feature].fillna('unknown')
        products[feature + '_encoded'] = encoder.fit_transform(products[feature])
        encoders[feature] = encoder

    products['price'] = products['price'].fillna(products['price'].median())
    products['price_normalized'] = (products['price'] - products['price'].mean()) / products['price'].std()

    product_embeddings = np.vstack([products[feature + '_encoded'].values for feature in product_features] + 
                                    [products['price_normalized'].values]).T
    return product_embeddings


if __name__ == "__main__":
    X_tensor, y_tensor, product_feature_tensor, product_encoder = preprocess_data(
        '/data/processed/sessions.csv', '/data/processed/products.csv')
