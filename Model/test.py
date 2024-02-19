import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
import numpy as np
from sklearn.metrics import ndcg_score
from torchgen.context import F


# Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Function to create DataLoader from CSV files
def create_data_loader(csv_file, batch_size=1024):
    df = pd.read_csv(csv_file)
    data_list = [Data(edge_index=torch.tensor([[row.userid], [row.itemid]], dtype=torch.long)) for index, row in df.iterrows()]
    return DataLoader(data_list, batch_size=batch_size, shuffle=True)

# Function to compute recall and NDCG
def evaluate_recall_ndcg(model, test_data, k=20):
    model.eval()
    all_scores = []
    all_labels = []  # Assuming labels are provided
    with torch.no_grad():
        for data in test_data:
            scores = model(data).squeeze().cpu().numpy()
            all_scores.extend(scores)
            # Assuming labels are 1 for all test data since it's positive
            all_labels.extend(np.ones(scores.shape))

    # Sort scores in descending order and compute recall and NDCG
    sorted_indices = np.argsort(-np.array(all_scores))
    sorted_labels = np.array(all_labels)[sorted_indices]
    recall_at_k = np.mean(sorted_labels[:k])
    true_relevance = np.zeros_like(sorted_labels)
    true_relevance[:k] = sorted_labels[:k]
    ndcg_at_k = ndcg_score([sorted_labels], [true_relevance])

    return recall_at_k, ndcg_at_k

# Initialize model, optimizer, and data loaders
model = GNNModel(in_channels=16, hidden_channels=32, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train_loader = create_data_loader('train_data.csv')
test_loader = create_data_loader('test_data.csv')

# Training loop
for epoch in range(20):  # Simplified training loop for illustration
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        # Here you should define your loss function based on the task
        loss = F.binary_cross_entropy_with_logits(out, torch.ones_like(out))  # Simplified loss for illustration
        loss.backward()
        optimizer.step()

# Evaluation
recall_at_k, ndcg_at_k = evaluate_recall_ndcg(model, test_loader, k=20)
print(f"Recall@20: {recall_at_k}, NDCG@20: {ndcg_at_k}")
