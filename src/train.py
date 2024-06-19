# src/train.py
"""
Script for training the BERT4Rec model.
"""

import torch
import tokenizers
import pickle
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
import os

from data_preprocessing import preprocess_data
from model import BERT4RecWithFeatures

def train_model_with_features(model, train_loader, optimizer, scheduler, num_epochs=3):
    """
    Train the BERT4Rec model with product features.

    Parameters:
    - model (nn.Module): BERT4Rec model.
    - train_loader (DataLoader): DataLoader for training data.
    - optimizer (Optimizer): Optimizer.
    - scheduler (Scheduler): Learning rate scheduler.
    - num_epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, product_features, labels = batch
            
            outputs = model(input_ids, product_features)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

def save_model(model, model_path, product_encoder):
    """
    Save the trained model and necessary components.

    Parameters:
    - model (nn.Module): The trained model.
    - model_path (str): Path to save the model.
    - product_encoder (LabelEncoder): Encoder for product IDs.
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Save model state_dict
    model_save_path = os.path.join(model_path, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_save_path)
    
    # Save product encoder
    encoder_save_path = os.path.join(model_path, 'product_encoder.pkl')
    with open(encoder_save_path, 'wb') as f:
        pickle.dump(product_encoder, f)
    

if __name__ == "__main__":
    # Load and preprocess data
    X_tensor, y_tensor, product_feature_tensor, product_encoder = preprocess_data(
        '/data/preprocess/sessions.csv', '/data/preprocess/products.csv')
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, product_feature_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Define model
    num_product_features = product_feature_tensor.shape[1]
    model = BERT4RecWithFeatures(num_product_features, num_labels=len(product_encoder.classes_))
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * 3  # Assuming 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Train model
    train_model_with_features(model, train_loader, optimizer, scheduler, num_epochs=3)
    
    # Save the model
    save_model(model, '/saved_models/bert4rec_model', product_encoder)
