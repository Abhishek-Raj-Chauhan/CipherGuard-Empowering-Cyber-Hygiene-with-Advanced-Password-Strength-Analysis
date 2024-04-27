# Importing necessary Libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import joblib

# Read the dataset
data = pd.read_csv('training.csv')

# Features and labels extraction
features = data['password'].astype(str)
labels = data['strength'].astype(int)

# Define classifiers
log_reg_classifier = LogisticRegression(max_iter=1000)
rf_classifier = RandomForestClassifier()
nb_classifier = MultinomialNB()
dt_classifier = DecisionTreeClassifier()

# Create a pipeline for each classifier
log_reg_pipeline = Pipeline([('tfidf', TfidfVectorizer(analyzer='char')), ('log_reg', log_reg_classifier)])
rf_pipeline = Pipeline([('tfidf', TfidfVectorizer(analyzer='char')), ('rf', rf_classifier)])
nb_pipeline = Pipeline([('tfidf', TfidfVectorizer(analyzer='char')), ('nb', nb_classifier)])
dt_pipeline = Pipeline([('tfidf', TfidfVectorizer(analyzer='char')), ('dt', dt_classifier)])

# Create the ensemble model with soft voting
soft_voting_classifier = VotingClassifier(estimators=[
    ('log_reg', log_reg_pipeline),
    ('rf', rf_pipeline),
    ('nb', nb_pipeline),
    ('dt', dt_pipeline)
], voting='soft')

# Create the ensemble model with hard voting
hard_voting_classifier = VotingClassifier(estimators=[
    ('log_reg', log_reg_pipeline),
    ('rf', rf_pipeline),
    ('nb', nb_pipeline),
    ('dt', dt_pipeline)
], voting='hard')

# Fit the models
soft_voting_classifier.fit(features, labels)
print('Soft Voting Training Accuracy: ', soft_voting_classifier.score(features, labels))
training_accuracy_soft = soft_voting_classifier.score(features, labels)
# Save model and training accuracy
model_with_accuracy_soft = {'model': soft_voting_classifier, 'accuracy': training_accuracy_soft}

hard_voting_classifier.fit(features, labels)
print('Hard Voting Training Accuracy: ', hard_voting_classifier.score(features, labels))
training_accuracy_hard = hard_voting_classifier.score(features, labels)
# Save model and training accuracy
model_with_accuracy_hard = {'model': hard_voting_classifier, 'accuracy': training_accuracy_hard}


# Save the models
joblib.dump(model_with_accuracy_soft, 'Soft_Voting_Ensemble_Model.joblib')
joblib.dump(model_with_accuracy_hard, 'Hard_Voting_Ensemble_Model.joblib')
