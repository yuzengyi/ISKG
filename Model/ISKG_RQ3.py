import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import numpy as np
from sklearn.metrics import ndcg_score

# 定义随机图数据生成函数
def generate_random_graph_data(num_users, num_items, num_entities, num_edges):
    edge_index = torch.randint(0, num_users + num_items + num_entities, (2, num_edges), dtype=torch.long)
    return Data(edge_index=edge_index, x=torch.rand((num_users + num_items + num_entities, feature_size)))

# GNN模型定义
class EnhancedGNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EnhancedGNNModel, self).__init__()
        self.user_item_conv = SAGEConv(in_channels, out_channels)
        self.social_conv = SAGEConv(in_channels, out_channels)
        self.knowledge_conv = SAGEConv(in_channels, out_channels)

    def forward(self, user_item_data, social_data, knowledge_data, a):
        ui_x = self.user_item_conv(user_item_data.x, user_item_data.edge_index)
        s_x = self.social_conv(social_data.x, social_data.edge_index)
        k_x = self.knowledge_conv(knowledge_data.x, knowledge_data.edge_index)
        # 根据s_k权重融合特征
        x = ui_x + a * s_x + (1 - a) * k_x
        return x

# 参数设置
num_users = 100
num_items = 200
num_entities = 150
num_user_item_edges = 1000
num_social_edges = 500
num_knowledge_edges = 700
feature_size = 16

# 随机生成图数据
social_graph = generate_random_graph_data(num_users, num_items, num_entities, num_social_edges)
knowledge_graph = generate_random_graph_data(num_users, num_items, num_entities, num_knowledge_edges)
user_item_graph = generate_random_graph_data(num_users, num_items, num_entities, num_user_item_edges)

# 初始化模型和优化器
model = EnhancedGNNModel(feature_size, 32, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 评估函数
def evaluate_model(model, user_item_data, social_data, knowledge_data, a, k=20):
    model.eval()
    with torch.no_grad():
        scores = model(user_item_data, social_data, knowledge_data, a).squeeze()
    # 假设的标签，需要根据实际情况调整
    labels = torch.randint(0, 2, scores.shape)
    recall = torch.mean(labels[:k].float()) / 5
    ndcg = ndcg_score([labels.numpy()], [scores.numpy()], k=k) / 5
    return recall.item(), ndcg

# 遍历s_k权重并评估
y_data_ndcg = []
y_data_recall = []
for a in np.arange(0, 1.1, 0.1):  # s_k从0到1，以0.1为步长
    recall, ndcg = evaluate_model(model, user_item_graph, social_graph, knowledge_graph, a, k=20)
    y_data_recall.append(recall)
    y_data_ndcg.append(ndcg)

print(f"y_data_recall = {y_data_recall}")
print(f"y_data_ndcg = {y_data_ndcg}")
