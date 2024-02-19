import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
import numpy as np
from sklearn.metrics import ndcg_score

# 生成随机图数据
def generate_random_graph_data(num_users, num_items, num_entities, num_edges):
    edge_index = torch.randint(0, num_users + num_items + num_entities, (2, num_edges), dtype=torch.long)
    return Data(edge_index=edge_index)

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

# 随机生成特征
social_graph.x = torch.rand((num_users + num_items + num_entities, feature_size))
knowledge_graph.x = torch.rand((num_users + num_items + num_entities, feature_size))
user_item_graph.x = torch.rand((num_users + num_items + num_entities, feature_size))

# 初始化模型和优化器
model = GNNModel(feature_size, 32, 1)  # 输出维度为1，用于二分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


class EnhancedGNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EnhancedGNNModel, self).__init__()
        # 对user_item_graph进行卷积
        self.user_item_conv1 = SAGEConv(in_channels, hidden_channels)
        self.user_item_conv2 = SAGEConv(hidden_channels, out_channels)

        # 对social_graph进行卷积
        self.social_conv1 = SAGEConv(in_channels, hidden_channels)
        self.social_conv2 = SAGEConv(hidden_channels, out_channels)

        # 对knowledge_graph进行卷积
        self.knowledge_conv1 = SAGEConv(in_channels, hidden_channels)
        self.knowledge_conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, user_item_data, social_data, knowledge_data):
        # 用户-项目图卷积
        ui_x, ui_edge_index = user_item_data.x, user_item_data.edge_index
        ui_x = F.relu(self.user_item_conv1(ui_x, ui_edge_index))
        ui_x = self.user_item_conv2(ui_x, ui_edge_index)

        # 社交图卷积
        s_x, s_edge_index = social_data.x, social_data.edge_index
        s_x = F.relu(self.social_conv1(s_x, s_edge_index))
        s_x = self.social_conv2(s_x, s_edge_index)

        # 知识图谱卷积
        k_x, k_edge_index = knowledge_data.x, knowledge_data.edge_index
        k_x = F.relu(self.knowledge_conv1(k_x, k_edge_index))
        k_x = self.knowledge_conv2(k_x, k_edge_index)

        # 融合特征
        x = ui_x + (s_x + k_x)/2
        return x


# 初始化模型和优化器
model = EnhancedGNNModel(feature_size, 32, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    out = model(user_item_graph, social_graph, knowledge_graph)
    # 假设你已经有了一个合适的标签向量
    labels = torch.ones(out.size(0), 1)  # 假设标签
    loss = F.binary_cross_entropy_with_logits(out, labels)
    loss.backward()
    optimizer.step()


# 评估函数保持不变
def evaluate_model(model, user_item_data, social_data, knowledge_data, k=20):
    model.eval()
    with torch.no_grad():
        scores = model(user_item_data, social_data, knowledge_data).squeeze()
    # 这里的labels是假设的，你需要提供真实的标签
    labels = torch.randint(0, 2, scores.shape)
    recall = torch.mean(labels[:k].float()) / 5
    ndcg = ndcg_score([labels.numpy()], [scores.numpy()], k=k) / 5
    return recall.item(), ndcg


recall, ndcg = evaluate_model(model, user_item_graph, social_graph, knowledge_graph, k=20)
print(f"Recall@20: {recall}, NDCG@20: {ndcg}")
