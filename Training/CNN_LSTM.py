import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Read the File
data = pd.read_csv('training.csv')

# Features which are passwords
features = data['password'].astype(str)

# Labels which are strength of password
labels = data['strength']

# Tokenize the passwords (features)
tokenizer = Tokenizer(char_level=True)  # Using character-level tokenization
tokenizer.fit_on_texts(features)
sequences = tokenizer.texts_to_sequences(features)
max_length = max(len(seq) for seq in sequences)  # Find max sequence length
features_processed = pad_sequences(sequences, maxlen=max_length, padding='post')  # Padding sequences

# Define the CNN-BiLSTM model for classification
def create_cnn_bilstm_model(input_shape, num_classes):
    model_input = Input(shape=input_shape)
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100)(model_input)
    conv1d_layer1 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding_layer)
    maxpooling_layer1 = MaxPooling1D(pool_size=2)(conv1d_layer1)
    conv1d_layer2 = Conv1D(filters=128, kernel_size=3, activation='relu')(maxpooling_layer1)
    maxpooling_layer2 = MaxPooling1D(pool_size=2)(conv1d_layer2)
    bilstm_layer = Bidirectional(LSTM(128))(maxpooling_layer2)
    output = Dense(num_classes, activation='softmax')(bilstm_layer)
    model = Model(inputs=model_input, outputs=output)
    return model

# Adjust the input shape and number of classes
input_shape = (features_processed.shape[1],)
num_classes = 3  # Assuming 3 classes for password strength
cnn_bilstm_model = create_cnn_bilstm_model(input_shape, num_classes)

# Compile the model with appropriate loss function for classification
cnn_bilstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = cnn_bilstm_model.fit(features_processed, labels, epochs=1, batch_size=64, validation_split=0.2)  

# Save the model
cnn_bilstm_model.save('cnn_bilstm_model.h5')

# Save the accuracy
accuracy = history.history['accuracy'][0]  # Get accuracy from training history
with open('cnn_bilstm_accuracy.txt', 'w') as f:
    f.write(str(accuracy))

# Predict labels using the trained model
predicted_labels = cnn_bilstm_model.predict(features_processed)
predicted_labels = np.argmax(predicted_labels, axis=1)  # Convert probabilities to class labels

# Generate confusion matrix
cm = confusion_matrix(labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Weak', 'Medium', 'Strong'], yticklabels=['Weak', 'Medium', 'Strong'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()