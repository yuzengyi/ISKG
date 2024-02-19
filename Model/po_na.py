import pandas as pd
import numpy as np

# Load datasets
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Assuming the max item ID for negative sampling
max_item_id = train_data['itemid'].max()

def generate_negative_samples(data, num_negatives=1):
    negative_samples = []
    user_item_set = set(zip(data['userid'], data['itemid']))
    for _, row in data.iterrows():
        user = row['userid']
        for _ in range(num_negatives):
            negative_item = np.random.randint(1, max_item_id + 1)
            while (user, negative_item) in user_item_set:
                negative_item = np.random.randint(1, max_item_id + 1)
            negative_samples.append([user, negative_item, 0])  # 0 indicates negative interaction
    return pd.DataFrame(negative_samples, columns=['userid', 'itemid', 'interaction'])

# Generate negative samples for training data
negative_samples = generate_negative_samples(train_data)

# Concatenate positive and negative samples
positive_data = train_data.assign(interaction=1)  # Ensure all training data is marked as positive
train_data_with_negatives = pd.concat([positive_data, negative_samples])

# Splitting test data (assuming test_data already only contains positive examples)
test_positive_data = test_data.assign(interaction=1)
test_negative_data = generate_negative_samples(test_data)

# Save the datasets if needed
train_data_with_negatives.to_csv('train_data_with_negatives.csv', index=False)
test_positive_data.to_csv('test_positive_data.csv', index=False)
test_negative_data.to_csv('test_negative_data.csv', index=False)

# Output file paths for convenience
file_paths = {
    'train_data_with_negatives': 'train_data_with_negatives.csv',
    'test_positive_data': 'test_positive_data.csv',
    'test_negative_data': 'test_negative_data.csv'
}
print(file_paths)
