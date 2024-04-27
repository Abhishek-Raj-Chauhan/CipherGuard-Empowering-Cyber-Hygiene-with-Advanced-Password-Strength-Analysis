import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# Read the dataset
data = pd.read_csv('training.csv')

# Introduce noise to labels
noise_percentage = 0.2  # 20% of labels will be flipped randomly
np.random.seed(42)  # For reproducibility
noise_indices = np.random.choice(data.index, size=int(len(data) * noise_percentage), replace=False)
data.loc[noise_indices, 'strength'] = np.random.randint(0, 3, size=len(noise_indices))

# Features (passwords) and labels (strength of passwords)
features = data['password'].astype('str')
labels = data['strength'].astype('int')

# Create pipeline
classifier_model = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char')),
                ('randomForest',RandomForestClassifier(n_estimators=100, max_depth=50, criterion='entropy')),
])

# Fit the model
classifier_model.fit(features, labels)

# Training Accuracy
print('Training Accuracy: ', classifier_model.score(features, labels))

# Calculate training accuracy
training_accuracy = classifier_model.score(features, labels)

# Save model and training accuracy
model_with_accuracy = {'model': classifier_model, 'accuracy': training_accuracy}
# Save model for RandomForestClassifier
joblib.dump(model_with_accuracy, 'RandomForest_Model.joblib')
