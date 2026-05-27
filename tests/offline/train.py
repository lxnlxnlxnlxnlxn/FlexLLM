from typing import List

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import argparse


def main(args: argparse.ArgumentParser):

    # Get parameters from command line
    model_name = args.model_name    # LLM model name
    stage = args.stage              # Inference stage: prefill or decode

    # Construct CSV data path (read ptime.csv or dtime.csv)
    file_path = f"/workspace/data/forward_time/{model_name}/{stage[0]}time.csv"
    
    # Load CSV dataset (bs: batch size, seql: sequence length, t(ms): time cost)
    df = pd.read_csv(file_path)
    
    # Input features: batch size and sequence length
    X = df[['bs', 'seql']]
    
    # Target label: inference time in milliseconds
    y = df['t(ms)']
    
    # Split dataset into training set (80%) and test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Initialize Random Forest regression model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Predict on test set
    rf_pred = rf_model.predict(X_test)

    # Print model training result and average error
    print(f"model = {model_name}, stage = {stage}")
    print(f"Mean Relative Error: {get_loss(rf_pred, y_test) * 100:.3f} %")

    # Save trained model to .pkl file for later inference prediction
    model_path = f"/workspace/data/forward_time/{model_name}/{stage}.pkl"
    joblib.dump(rf_model, model_path)
    
    print(f"Model saved to: {model_path}\n")


def get_loss(pred_val_list: List[float], true_val_list: List[float]) -> float:
    loss_list = list()
    for pred_val, true_val in zip(pred_val_list, true_val_list):
        loss = abs(pred_val - true_val) / true_val
        loss_list.append(loss)
    return sum(loss_list) / len(loss_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train time prediction model for FlexLLM")

    parser.add_argument(
        '--model_name', 
        type=str,
        choices=["opt-13b", "llama-1-13b", "llama-2-13b", "llama-3-8b"], 
        required=True, 
        help="Name of the LLM model to train"
    )
    parser.add_argument(
        '--stage', 
        type=str,
        choices=["prefill", "decode"], 
        required=True, 
        help="Inference stage: prefill or decode"
    )

    args = parser.parse_args()
    main(args=args)
    