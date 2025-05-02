import matplotlib.pyplot as plt
import pandas as pd
import torch
from model import RNNFallPredictor
from infer import load_and_prepare_csv

SEQ_LEN = 20
INPUT_SIZE = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_predictions(csv_path, model_path):
    model = RNNFallPredictor()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    inputs = load_and_prepare_csv(csv_path).to(DEVICE)
    outputs = model(inputs)
    probs = torch.sigmoid(outputs).squeeze().detach().cpu().numpy()

    # Extend timeline to match original data length (pad with first few values as NaN)
    timeline = [None] * (SEQ_LEN - 1) + probs.tolist()

    df = pd.read_csv(csv_path)
    true_labels = df.iloc[:, -1].values  # assuming last column is label

    # Align labels (last frame of each window)
    aligned_labels = [None] * (SEQ_LEN - 1) + true_labels[SEQ_LEN - 1:].tolist()

    plt.figure(figsize=(14, 5))
    plt.plot(timeline, label='Predicted Fall Probability', color='blue')
    plt.plot(aligned_labels, label='True Label (Fall=1)', linestyle='--', color='red', alpha=0.6)
    plt.axhline(0.5, color='gray', linestyle=':', label='Decision Threshold')
    plt.xlabel("Time Step")
    plt.ylabel("Fall Probability")
    plt.title("Fall Prediction Timeline")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_predictions(
        csv_path="../data/g1_traj_1/some_file.csv",     # Update path
        model_path="rnn_fall_model.pt"
    )
