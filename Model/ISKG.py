import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
import numpy as np
from sklearn.metrics import ndcg_score



# Load graph function remains the same
def load_graph(csv_file, edge_columns, user_offset=None, num_nodes=None):
    df = pd.read_csv(csv_file)
    edge_index = torch.tensor([df[edge_columns[0]].values, df[edge_columns[1]].values], dtype=torch.long)
    if user_offset:
        edge_index[0, :] += user_offset
    data = Data(edge_index=edge_index)
    if num_nodes:
        data.num_nodes = num_nodes
    return data

# GNN Model Definition
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

# Function to create a DataLoader from positive and negative samples
def create_data_loader(positive_csv, negative_csv, batch_size=1024):
    pos_df = pd.read_csv(positive_csv)
    neg_df = pd.read_csv(negative_csv)
    # Assuming the positive and negative CSVs have columns 'userid' and 'itemid'
    pos_data_list = [Data(edge_index=torch.tensor([[row.userid], [row.itemid]], dtype=torch.long)) for index, row in pos_df.iterrows()]
    neg_data_list = [Data(edge_index=torch.tensor([[row.userid], [row.itemid]], dtype=torch.long)) for index, row in neg_df.iterrows()]
    data_list = pos_data_list + neg_data_list
    return DataLoader(data_list, batch_size=batch_size, shuffle=True)

# Function to compute recall and NDCG
def evaluate_recall_ndcg(model, positive_data, negative_data, k=20):
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for data in positive_data:
            scores = model(data).squeeze().cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(np.ones(scores.shape))
        for data in negative_data:
            scores = model(data).squeeze().cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(np.zeros(scores.shape))

    # Assuming all_scores is a list of predicted scores and all_labels is the ground truth (1 for positive, 0 for negative)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Sort scores and labels together in descending order of scores
    sorted_indices = np.argsort(-all_scores)
    sorted_labels = all_labels[sorted_indices]

    # Compute recall@k
    recall_at_k = np.mean(sorted_labels[:k])

    # Compute NDCG@k
    true_relevance = np.zeros_like(sorted_labels)
    true_relevance[:k] = sorted_labels[:k]
    ndcg_at_k = ndcg_score([sorted_labels], [true_relevance])

    return recall_at_k, ndcg_at_k

# Assuming you have instantiated and trained your model as before
# Now using the create_data_loader function to prepare DataLoader instances
train_loader = create_data_loader('test_positive_data.csv', 'test_negative_data.csv')
test_positive_loader = DataLoader([Data(edge_index=torch.tensor([[row.userid], [row.itemid]], dtype=torch.long)) for index, row in pd.read_csv('test_positive_data.csv').iterrows()], batch_size=1024)
test_negative_loader = DataLoader([Data(edge_index=torch.tensor([[row.userid], [row.itemid]], dtype=torch.long)) for index, row in pd.read_csv('test_negative_data.csv').iterrows()], batch_size=1024)
# Define the model
model = GNNModel(in_channels=16, hidden_channels=32, out_channels=1)  # Adjust the channels according to your data

# Initialize the optimizer (assuming you're using Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop (simplified)
for epoch in range(200):  # Number of epochs
    model.train()  # Set the model to training mode
    for data in train_loader:  # Assuming train_loader is a DataLoader instance
        optimizer.zero_grad()  # Clear gradients
        out = model(data)  # Forward pass
        loss = ...  # Compute your loss based on the task
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters

# Example evaluation
recall_at_k, ndcg_at_k = evaluate_recall_ndcg(model, test_positive_loader, test_negative_loader, k=20)
print(f"Recall@20: {recall_at_k}, NDCG@20: {ndcg_at_k}")
