import torch
import torch.nn as nn
import pandas as pd
from model import RNNFallPredictor

SEQ_LEN = 20
INPUT_SIZE = 12  # change if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_csv(file_path):
    df = pd.read_csv(file_path)
    features = df.iloc[:, :INPUT_SIZE].values

    sequences = []
    for i in range(len(features) - SEQ_LEN):
        seq = features[i:i+SEQ_LEN]
        sequences.append(seq)

    return torch.tensor(sequences, dtype=torch.float32)

def predict(model_path, csv_path):
    model = RNNFallPredictor()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    inputs = load_and_prepare_csv(csv_path).to(DEVICE)
    outputs = model(inputs)
    probs = torch.sigmoid(outputs).squeeze().detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)

    for i, (p, prob) in enumerate(zip(preds, probs)):
        print(f"Window {i+1}: {'FALL' if p else 'SAFE'} (prob: {prob:.2f})")

if __name__ == "__main__":
    csv_to_test = "../data/g1_traj_1/some_file.csv"  # change as needed
    model_file = "rnn_fall_model.pt"
    predict(model_file, csv_to_test)
