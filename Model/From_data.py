import pandas as pd
import numpy as np

# Define the size of each dataset
num_users = 500
num_items = 300
num_entities = 450
num_interactions = 2000
num_social_links = 1500
num_knowledge_triplets = 2500

# User-Item Interaction Graph
user_item_interactions = pd.DataFrame({
    'userid': np.random.choice(range(1, num_users + 1), num_interactions, replace=True),
    'itemid': np.random.choice(range(1, num_items + 1), num_interactions, replace=True),
    'I': np.random.choice([0, 1], num_interactions, replace=True)
})

# Social Graph
social_graph = pd.DataFrame({
    'userid': np.random.choice(range(1, num_users + 1), num_social_links, replace=True),
    'friendsid': np.random.choice(range(1, num_users + 1), num_social_links, replace=True)
})

# Knowledge Graph
knowledge_graph = pd.DataFrame({
    'itemid': np.random.choice(range(1, num_items + 1), num_knowledge_triplets, replace=True),
    'categoriesid': np.random.choice(range(1, num_entities + 1), num_knowledge_triplets, replace=True)
})

# Split User-Item Interaction Graph into train, valid, and test sets
train_size = int(0.8 * num_interactions)
valid_size = int(0.1 * num_interactions)
test_size = num_interactions - train_size - valid_size

train_data = user_item_interactions[:train_size]
valid_data = user_item_interactions[train_size:train_size + valid_size]
test_data = user_item_interactions[train_size + valid_size:]

# Save the datasets to CSV files
user_item_interactions.to_csv('user_item_interactions.csv', index=False)
social_graph.to_csv('social_graph.csv', index=False)
knowledge_graph.to_csv('knowledge_graph.csv', index=False)
train_data.to_csv('train_data.csv', index=False)
valid_data.to_csv('valid_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
