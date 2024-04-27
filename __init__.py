from flask import Flask, render_template, flash, request
import joblib
import scipy
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score  # Add this import statement
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
from passTime import evaluate_password_strength, crack_speed, format_time,custom_scaling
matplotlib.use('Agg')  # Use the 'Agg' backend


app = Flask(__name__)


def load_cnn_bilstm_model(model_path, accuracy_path):
    cnn_bilstm_model = load_model(model_path)
    cnn_bilstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    with open(accuracy_path, 'r') as f:
        accuracy = float(f.read())
    return cnn_bilstm_model, accuracy
def load_siamese_model(model_path, accuracy_path):
    siamese_model = load_model(model_path)
    siamese_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    with open(accuracy_path, 'r') as f:
        accuracy = float(f.read())
    return siamese_model, accuracy


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/main/', methods=['GET', 'POST'])
def mainpage():
    data = pd.read_csv('training.csv')
    features = data['password'].astype(str)
   
    # Fit tokenizer on the data
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(features)
   
   
    if request.method == "POST":
        enteredPassword = request.form['password']
    else:
        return render_template('index.html')
   
     # Preprocess the input password
    sequence = tokenizer.texts_to_sequences([enteredPassword])
    processed_password = pad_sequences(sequence, maxlen=220, padding='post')
   
    # Load the algorithm models using joblib
    DecisionTree_Model_withAccu = joblib.load('DecisionTree_Model.joblib')
    DecisionTree_Model = DecisionTree_Model_withAccu['model']
   
    LogisticRegression_Model_withAccu = joblib.load('LogisticRegression_Model.joblib')
    LogisticRegression_Model = LogisticRegression_Model_withAccu['model']
   
    NaiveBayes_Model_withAccu = joblib.load('NaiveBayes_Model.joblib')
    NaiveBayes_Model = NaiveBayes_Model_withAccu['model']
   
    RandomForest_Model_withAccu = joblib.load('RandomForest_Model.joblib')
    RandomForest_Model = RandomForest_Model_withAccu['model']
   
    NeuralNetwork_Model_withAccu = joblib.load('NeuralNetwork_Model.joblib')
    NeuralNetwork_Model = NeuralNetwork_Model_withAccu['model']


    EnsembelSoft_Model_withAccu = joblib.load('Soft_Voting_Ensemble_Model.joblib')
    EnsembelSoft_Model = EnsembelSoft_Model_withAccu['model']
   
    EnsembelHard_Model_withAccu = joblib.load('Hard_Voting_Ensemble_Model.joblib')
    EnsembelHard_Model = EnsembelHard_Model_withAccu['model']
   
    cnn_bilstm_model, cnn_bilstm_accuracy = load_cnn_bilstm_model('cnn_bilstm_model.h5', 'cnn_bilstm_accuracy.txt')
    siamese_model, siamese_accuracy = load_siamese_model('transformer_model.h5', 'transformer_accuracy.txt')
    Password = [enteredPassword]


    # Predict the strength
    DecisionTree_Test = DecisionTree_Model.predict(Password)
    LogisticRegression_Test = LogisticRegression_Model.predict(Password)
    NaiveBayes_Test = NaiveBayes_Model.predict(Password)
    RandomForest_Test = RandomForest_Model.predict(Password)
    NeuralNetwork_Test = NeuralNetwork_Model.predict(Password)
    EnsembelSoft_Test = EnsembelSoft_Model.predict(Password)
    EnsembelHard_Test = EnsembelHard_Model.predict(Password)
    cnn_bilstm_Test = cnn_bilstm_model.predict(processed_password)
    predicted_bil_strength = np.argmax(cnn_bilstm_Test)
    siamese_Test = siamese_model.predict(processed_password)
    predicted_siamese_strength = np.argmax(siamese_Test)
   
    # Calculate the time to crack the password
    entropy = evaluate_password_strength(enteredPassword)
    length_factor = custom_scaling(enteredPassword)
    cracked_seconds = ((entropy ** length_factor) /crack_speed)  # Time in seconds
    time_value, time_unit = format_time(cracked_seconds)
    if time_unit in ['seconds', 'minutes', 'hours', 'days', 'years']:
        time_unit = time_unit + ' (Vulnerable)'
    #accuracy
    # Load the actual labels
    labels = data['strength'].astype('int')
    actual_labels = labels[1]  # the label for the entered password is the first one in the dataset
    accuracyDecision = DecisionTree_Model_withAccu['accuracy']*100
    accuracyLogistic = LogisticRegression_Model_withAccu['accuracy']*100
    accuNaive = NaiveBayes_Model_withAccu['accuracy']*100
    accuRandFor = RandomForest_Model_withAccu['accuracy']*100
    accuNeuralmlp = NeuralNetwork_Model_withAccu['accuracy']*100
    accuEnSoft = EnsembelSoft_Model_withAccu['accuracy']*100
    accuEnHard = EnsembelHard_Model_withAccu['accuracy']*100
    acculstm = cnn_bilstm_accuracy*100
    accutr = siamese_accuracy*100
   
   # Plot predictions of each algorithm
    algorithms = ['Decision Tree', 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'Neural Network', 'Ensemble Soft', 'Ensemble Hard', 'CNN BiLSTM', 'siamese']
    predictions = [DecisionTree_Test[0], LogisticRegression_Test[0], NaiveBayes_Test[0], RandomForest_Test[0], NeuralNetwork_Test[0], EnsembelSoft_Test[0], EnsembelHard_Test[0], predicted_bil_strength, predicted_siamese_strength]


    # Count the number of predictions for each category
    weak_count = predictions.count(0)
    medium_count = predictions.count(1)
    strong_count = predictions.count(2)


    # Create labels for the pie chart
    labels = ['Weak', 'Medium', 'Strong']
    sizes = [weak_count, medium_count, strong_count]


    # Set colors for each category
    colors = ['red', 'orange', 'green']


    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Predicted Password Strengths')


    # Add legend
    plt.legend(labels, loc="best")


    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    graph_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()    
   
    return render_template("main.html", DecisionTree=DecisionTree_Test,
    LogReg=LogisticRegression_Test,
    NaiveBayes=NaiveBayes_Test,
    RandomForest=RandomForest_Test,
    NeuralNetwork=NeuralNetwork_Test,
    EnsembelSoft=EnsembelSoft_Test,
    EnsembelHard=EnsembelHard_Test,
    CnnBilstm=predicted_bil_strength,
    Transformer=predicted_siamese_strength,
    accDec = accuracyDecision,
    accuLog = accuracyLogistic,
    accuNav = accuNaive,
    accuRF = accuRandFor,
    accuNeu = accuNeuralmlp,
    accues = accuEnSoft,
    accueh = accuEnHard,
    acculs = acculstm,
    accutr=accutr,
    histogram=graph_base64,
    tval = time_value,
    tunit=time_unit
    )


if __name__ == "__main__":
    app.run()
