# BERT4Rec for Amazon KDD Cup 2023

This repository implements a next-item prediction system using the BERT4Rec model, applied to the Amazon KDD Cup 2023 dataset. The system predicts the next item a user is likely to buy based on their purchase history and the features of the products.

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

This project demonstrates the use of BERT4Rec for next-item prediction on a shopping dataset. The BERT4Rec model leverages the power of transformers to provide sequence-based recommendations by learning patterns from user sessions and product features.

## Project Structure

- **src/**: Source code for the project including data processing, model definition, training, evaluation, and loading.
  - `data_preprocessing.py`: Script for preprocessing session and product data, converting them into tensors for model input.
  - `model.py`: Contains the BERT4Rec model definition.
  - `train.py`: Training script that loads data, trains the BERT4Rec model, and saves the trained model.
  - `evaluate.py`: Script for evaluating the trained model’s performance.
  - `load_model.py`: Script for loading the trained model and necessary components for inference or further training.

- **notebooks/**: Jupyter Notebooks for exploratory data analysis (EDA).
  - `exploratory_data_analysis.ipynb`: Notebook containing cleaning and analysis of the dataset to gain insights and validate preprocessing steps.

- **saved_models/**: Directory to save and store trained models and their components.
  - `bert4rec_model/`: Directory where the trained BERT4Rec model and related files (such as encoders and tokenizers) are saved.

- **README.md**: The main README file that provides an overview and documentation for the project.

- **requirements.txt**: List of required Python packages and their versions needed to run the project.


## Acknowledgments

This project is based on the data provided for the Amazon KDD Cup 2023. The implementation utilizes the BERT4Rec architecture, a powerful transformer-based model for sequential recommendation.

## License

This project is licensed under the MIT License.
