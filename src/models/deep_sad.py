import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data.data_utils import load_standardized_feature_set


class DeepSADNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, rep_dim=32):
        super(DeepSADNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim)
        )

    def forward(self, x):
        return self.encoder(x)


def train_deep_sad(model, num_epochs=300, lr=1e-3, device='cpu'):
    dataloader = create_data_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    # Initialize center c
    c = torch.zeros(model.encoder[-1].out_features, device=device)

    # Estimate c from a forward pass
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            c += outputs.sum(dim=0)
    c /= len(dataloader.dataset)
    c[(abs(c) < 1e-6)] = 1e-6  # Avoid trivial solutions

    # Train
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)

            # Loss
            normal_loss = dist[labels == 0].mean()
            anomaly_loss = (-dist[labels == 1]).mean()  # Push anomalies away
            # Sometimes we don't have any labeled anomalies. In these cases we compute nan loss and fill w/
            # 0
            print(normal_loss, anomaly_loss)
            if  dist[labels == 0].mean() >  dist[labels == 1].mean():
                normal_loss *= 50

            # anomaly_loss = torch.where(
            #     torch.isnan(anomaly_loss), 
            #     torch.tensor(0.0, device=anomaly_loss.device),
            #     anomaly_loss
            # )
            # anomaly_loss *= 10
            margin = 20.0  # or tune this (5.0, 20.0)

            # # Anomaly loss: only penalize if anomaly is too close
            # if dist[labels].numel() > 0:
            #     anomaly_loss = torch.mean(torch.clamp(margin - dist[labels], min=0))
            # else:
            #     anomaly_loss = torch.tensor(0.0, device=device)

            loss = normal_loss + anomaly_loss

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
        for inputs, _ in create_data_loader(resample=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            scores.append(dist.cpu())
    
    scores = torch.cat(scores)
    df = load_standardized_feature_set(include_known=True)
    df["anomaly_score"] = scores.numpy()
    threshold = df["anomaly_score"].quantile(quantile)
    df["anomaly_label"] = 1
    df.loc[df["anomaly_score"] > threshold, "anomaly_label"] = -1
    return df

def create_data_loader(resample = True):
    df = load_standardized_feature_set(include_known=True)

    insiders = df[df['known_instance'] == 1]
    normals = df[df['known_instance'] == 0]

    if resample: 
        # Oversample insiders: sample with replacement
        oversample_factor = 5  # Try 5x more insider trades
        insiders_oversampled = insiders.sample(
            n=oversample_factor * len(insiders), 
            replace=True, 
            random_state=42
        )

        # Combine oversampled insiders with normals
        new_df = pd.concat([normals, insiders_oversampled]).sample(frac=1, random_state=42)
    else:
        new_df = df

    # Convert to tensor
    X = torch.tensor(new_df.drop(columns=['known_instance']).values, dtype=torch.float32)
    y = torch.tensor(new_df['known_instance'].values, dtype=torch.long)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader


def run_deep_sad(quantile):
    model = DeepSADNet(input_dim=19)
    model, c =  train_deep_sad(model)
    return get_anomaly_scores(model, c, quantile)
