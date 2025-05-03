import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data.data_utils import load_standardized_feature_set


class DeepSVDDNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, rep_dim=32):
        super(DeepSVDDNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim)
        )

    def forward(self, x):
        return self.encoder(x)


def train_deep_svdd(model, num_epochs=300, lr=1e-3, device='cpu'):
    dataloader = create_data_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    # Initialize center c
    c = torch.zeros(model.encoder[-1].out_features, device=device)

    # Estimate c from a forward pass
    model.eval()
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            c += outputs.sum(dim=0)
    c /= len(dataloader.dataset)
    c[(abs(c) < 1e-6)] = 1e-6  # Avoid trivial solutions

    # Train
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in dataloader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            loss = torch.mean(torch.sum((outputs - c) ** 2, dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')
    
    return model, c


def get_anomaly_scores(model, c, quantile, device='cpu'):
    model.eval()
    scores = []

    with torch.no_grad():
        for inputs in create_data_loader():
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            scores.append(dist.cpu())
    
    scores = torch.cat(scores)
    df = load_standardized_feature_set(include_known=False)
    df["anomaly_score"] = scores.numpy()
    threshold = df["anomaly_score"].quantile(quantile)
    df["anomaly_label"] = 1
    df.loc[df["anomaly_score"] > threshold, "anomaly_label"] = -1
    return df

def create_data_loader():
    df = load_standardized_feature_set(include_known=False)
    X = torch.tensor(df.values, dtype=torch.float32)
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader


def run_deep_svdd(quantile):
    model = DeepSVDDNet(input_dim=19)
    model, c =  train_deep_svdd(model)
    return get_anomaly_scores(model, c, quantile)
