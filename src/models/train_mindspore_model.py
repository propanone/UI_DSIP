# train_mindspore_model.py

import os
import time
import argparse
import pandas as pd
import numpy as np
import mindspore as ms
from mindspore import context

# Import the training function from our training module
from mindspore_mlp_training import train_mindspore_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MindSpore MLP model for client risk prediction')
    parser.add_argument('--data', type=str, default='data.csv', help='Path to the CSV data file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU', 'Ascend'], 
                        help='Device target for MindSpore')
    
    args = parser.parse_args()
    
    # Set MindSpore context
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    
    print(f"Starting MindSpore MLP model training with {args.epochs} epochs on {args.device}...")
    print(f"Using data file: {args.data}")
    
    start_time = time.time()
    
    # Train the model
    model, scaler = train_mindspore_model()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print("Model and preprocessors have been saved and are ready to use in the Streamlit app.")