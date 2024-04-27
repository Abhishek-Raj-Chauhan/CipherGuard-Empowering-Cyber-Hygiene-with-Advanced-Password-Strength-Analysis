import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import joblib

# Read the dataset
print("Reading the dataset...")
data = pd.read_csv('modified_dataset.csv')
print("Dataset read successfully. Number of entries:", len(data))

# Introduce noise to labels
noise_percentage = 0.1  # 10% of labels will be flipped randomly
np.random.seed(42)  # For reproducibility
noise_indices = np.random.choice(data.index, size=int(len(data) * noise_percentage), replace=False)
data.loc[noise_indices, 'strength'] = np.random.randint(0, 3, size=len(noise_indices))

# Function to add noise to passwords
def add_noise(password):
    noise_percentage = 0.1  # 10% chance of adding noise to each password
    if np.random.rand() < noise_percentage:
        password_list = list(password)
        np.random.shuffle(password_list)
        return ''.join(password_list)
    else:
        return password

# Apply noise to each password
data['password'] = data['password'].apply(add_noise)

# Features (passwords) and labels (strength of passwords)
features = data['password'].astype('str')
labels = data['strength'].astype('int')

# Create pipeline
classifier_model = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char')),
                ('decisionTree',DecisionTreeClassifier()),
])

# Fit the model
classifier_model.fit(features, labels)
print("Model fitting completed.")

# Training Accuracy
print('Training Accuracy: ', classifier_model.score(features, labels))

# Calculate training accuracy
training_accuracy = classifier_model.score(features, labels)

# Save model and training accuracy
model_with_accuracy = {'model': classifier_model, 'accuracy': training_accuracy}
joblib.dump(model_with_accuracy, 'DecisionTree_Model.joblib')
