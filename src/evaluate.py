# evaluate.py
"""
Script for evaluating the BERT4Rec model.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.metrics import accuracy_score 

from data_preprocessing import preprocess_data
from model import BERT4RecWithFeatures

def evaluate_model_with_features(model, data_loader):
    """
    Evaluate the BERT4Rec model.

    Parameters:
    - model (nn.Module): BERT4Rec model.
    - data_loader (DataLoader): DataLoader for evaluation data.

    Returns:
    - accuracy (float): Evaluation accuracy.
    """
    model.eval()
    all_predictions, all_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids, product_features, labels = batch
           
            outputs = model(input_ids, product_features)
            predictions = torch.argmax(outputs, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

if __name__ == "__main__":
    # Load and preprocess data
    X_tensor, y_tensor, product_feature_tensor, product_encoder = preprocess_data(
        '/data/processed/sessions.csv', '/data/processed/products.csv')
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, product_feature_tensor, y_tensor)
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load the model
    num_product_features = product_feature_tensor.shape[1]
    model = BERT4RecWithFeatures(num_product_features, num_labels=len(product_encoder.classes_))
    model.load_state_dict(torch.load('/saved_models/bert4rec_model/pytorch_model.bin'))

    # Evaluate model
    accuracy = evaluate_model_with_features(model, eval_loader)
    print(f"Evaluation Accuracy: {accuracy:.2f}")
