# Import the necessary Libraries
import pandas as pd

# For text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# For creating a pipeline
from sklearn.pipeline import Pipeline

# Classifier Model (Naive Bayes)
from sklearn.naive_bayes import BernoulliNB

# To save the trained model on local storage
import joblib

# Read the File
data = pd.read_csv('training.csv')

# Features which are passwords
features = data.values[:, 1].astype('str')

# Labels which are strength of password
labels = data.values[:, -1].astype('int')

# Sequentially apply a list of transforms and a final estimator
classifier_model = Pipeline([
                ('tfidf', TfidfVectorizer(analyzer='char')),
                ('bernoulliNB',BernoulliNB()),
])

# Fit the Model
classifier_model.fit(features, labels)

# Training Accuracy
print('Training Accuracy: ',classifier_model.score(features, labels))
# Calculate training accuracy
training_accuracy = classifier_model.score(features, labels)

# Save model and training accuracy
model_with_accuracy = {'model': classifier_model, 'accuracy': training_accuracy}

# Save model for Logistic Regression
joblib.dump(model_with_accuracy, 'NaiveBayes_Model.joblib')