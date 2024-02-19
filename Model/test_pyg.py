import torch
from torch_geometric.data import Data, DataLoader
import pandas as pd
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from sklearn.metrics import ndcg_score

# def load_graph(csv_file, edge_columns, user_offset=None, num_nodes=None):
#     df = pd.read_csv(csv_file)
#     edge_index = torch.tensor([df[edge_columns[0]].values, df[edge_columns[1]].values], dtype=torch.long)
#     if user_offset:
#         edge_index[0, :] += user_offset  # Offset user IDs
#     data = Data(edge_index=edge_index)
#     if num_nodes:
#         data.num_nodes = num_nodes  # Specify the number of nodes if known
#     return data
def load_graph(csv_file, edge_columns, user_offset=None, num_nodes=None):
    df = pd.read_csv(csv_file)
    edge_index = torch.tensor([df[edge_columns[0]].values, df[edge_columns[1]].values], dtype=torch.long)
    if user_offset:
        edge_index[0, :] += user_offset  # Adjust for user ID offset if necessary

    # Create random features for each node if num_nodes is specified
    if num_nodes:
        x = torch.rand(num_nodes, 16)  # 假设 node_features 是 16
  # Replace feature_dimension with the size of your features
    else:
        x = None

    return Data(edge_index=edge_index, x=x)


# Assuming we know the maximum IDs for users, items, and entities
max_user_id = 500
max_item_id = 300
max_entity_id = 450

# Total number of nodes = max user ID + max item ID + max entity ID
total_nodes = max_user_id + max_item_id + max_entity_id

# Initialize random features for each node
node_features = torch.rand((total_nodes, 16))  # Let's assume 16-dimensional features

# Load graphs
social_graph = load_graph('social_graph.csv', edge_columns=['userid', 'friendsid'], user_offset=100000, num_nodes=total_nodes)
knowledge_graph = load_graph('knowledge_graph.csv', edge_columns=['itemid', 'categoriesid'], num_nodes=total_nodes)
user_item_graph = load_graph('user_item_interactions.csv', edge_columns=['userid', 'itemid'], num_nodes=total_nodes)

# Add features to graphs
social_graph.x = node_features
knowledge_graph.x = node_features
user_item_graph.x = node_features

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

# Initialize the model
model = GNNModel(in_channels=16, hidden_channels=32, out_channels=1)  # Output is 1-dimensional for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Example training loop (You need to define your own loss function based on your task)
for epoch in range(200):
    optimizer.zero_grad()
    out = model(user_item_graph)  # Use the user-item graph for training
    # Compute your loss here; you might need labels from your data
    # loss.backward()
    optimizer.step()

# Example evaluation function (You need to adjust it based on your task and data)
def evaluate_model(model, data, k=20):
    model.eval()
    with torch.no_grad():
        scores = model(data)
        # Sort scores and take top-k; this part depends on your data format
        # Calculate recall and NDCG; you might need true labels for these calculations
    return recall_at_k, ndcg_at_k

# You need to prepare your test_data as a PyG Data object with features
# recall, ndcg = evaluate_model(model, test_data, k=20)
# print(f"Recall@20: {recall}, NDCG@20: {ndcg}")
