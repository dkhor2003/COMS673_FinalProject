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
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

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
    # X = torch.tensor(np.random.randn(10, 10, 55))  # Shape: (3,2)
    # y = torch.tensor(np.random.randn(10, 1))  # Shape: (3,)

    # print(X.shape, y.shape)  # Should both have the same first dimension

    # dataset = TensorDataset(X, y)  # Should work correctly
    data_dir = "data/g1_traj"
    X, y = process_csv_into_dataset(data_dir)
    num_data = X.shape[0]
    num_features = X.shape[2]
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    print(X.size())
    print(y.size())
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * num_data)
    test_size = num_data - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Define model
    model = FalloverPredictor(input_size=num_features)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train and test the model
    train_model(model, train_loader, criterion, optimizer, epochs=20)
    test_model(model, test_loader)
