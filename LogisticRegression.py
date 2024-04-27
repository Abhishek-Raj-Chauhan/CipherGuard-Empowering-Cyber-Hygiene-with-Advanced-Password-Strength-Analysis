# Import the necessary packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Read the File
data = pd.read_csv('training.csv')

# Features which are passwords
features = data['password'].astype(str)

# Labels which are strength of password
labels = data['strength'].astype(int)

# Create a pipeline with feature scaling and logistic regression
classifier_model = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char')),
    ('scaler', StandardScaler(with_mean=False)),  # Fix here
    ('logisticRegression', LogisticRegression(multi_class='multinomial', solver='sag', max_iter=1000)),
])

# Fit the Model
classifier_model.fit(features, labels)

# Training Accuracy
print('Training Accuracy: ',classifier_model.score(features, labels))
# Calculate training accuracy
training_accuracy = classifier_model.score(features, labels)

# Save model and training accuracy
model_with_accuracy = {'model': classifier_model, 'accuracy': training_accuracy}
# Save the model for Logistic Regression
joblib.dump(model_with_accuracy, 'LogisticRegression_Model.joblib')
