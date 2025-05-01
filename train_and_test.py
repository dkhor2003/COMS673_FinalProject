from preprocessing import *
from nn import FalloverPredictor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}",flush=True)

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    
    traj_dir = "data/g1_traj_3"
    weights_dir = "nn_weights/lstm"
    X_file = "data/processed_data/X3.npy"
    y_file = "data/processed_data/y3.npy"
    
    if os.path.isfile(X_file) and os.path.isfile(y_file): # Used already saved processed X and y
        print("Loading saved features and targets")
        X, y = np.load(X_file), np.load(y_file)
    else:
        print("Generating features and targets")
        X, y = process_csv_into_dataset(traj_dir) # Otherwise generate and save them from the trajectories
        np.save(X_file, X)
        np.save(y_file, y)
        
    print("Before upsampling, number of 0 class: ", np.count_nonzero(y), " out of a total of ", len(y), " samples") 
    
    # Upsampling
    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]
    num_to_add = len(idx_class_0) - len(idx_class_1)
    idx_upsampled = np.random.choice(idx_class_1, size=num_to_add, replace=True)
    new_X_sample = X[idx_upsampled]
    new_y_sample = y[idx_upsampled]
    noise = np.random.uniform(low=-0.01, high=0.01, size=new_X_sample.shape)
    new_X_sample += noise
    X = np.concatenate([X, new_X_sample], axis=0)
    y = np.concatenate([y, new_y_sample], axis=0)
    
    print("After upsampling, number of 0 class: ", np.count_nonzero(y), " out of a total of ", len(y), " samples") 
    
    num_data = X.shape[0]
    num_features = X.shape[2]
    
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_data)
    test_size = num_data - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Define model
    model = FalloverPredictor(input_size=num_features)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Train and test the model
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    test_model(model, test_loader)
    
    torch.save(model.state_dict(), f"{weights_dir}/experiment3_weights.pth")
