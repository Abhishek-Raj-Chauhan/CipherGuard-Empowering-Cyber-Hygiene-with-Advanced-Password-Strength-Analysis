import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('modified_training.csv')
df = pd.DataFrame(data)
# Introduce label noise
noise_percentage = 0.1  # 10% of labels will be flipped randomly
np.random.seed(42)  # For reproducibility
noise_indices = np.random.choice(data.index, size=int(len(data) * noise_percentage), replace=False)
data.loc[noise_indices, 'strength'] = np.random.randint(0, 3, size=len(noise_indices))

# Function to shuffle characters within a password
def add_noise(password):
    if np.random.rand() < noise_percentage:
        password_list = list(password)
        np.random.shuffle(password_list)
        return ''.join(password_list)
    else:
        return password

# Apply noise to each password
df['password'] = df['password'].apply(add_noise)

# Save the modified DataFrame to a new CSV file
df.to_csv('modified_training2.csv', index=False)

print("Modified dataset saved as modified_training.csv")
