import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, Dropout, Dense, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the PositionalEncoding layer directly
def positional_encoding(maxlen, d_model):
    angle_rates = 1 / np.power(10000, (2 * np.arange(d_model) / np.float32(d_model)))
    position = np.arange(maxlen)[:, np.newaxis]
    angle_rads = position * angle_rates[np.newaxis, :]
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Apply sin to even indices
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Apply cos to odd indices
    return angle_rads[np.newaxis, ...]

# Define the point-wise feed forward network
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        Dense(dff, activation='relu'),  # Hidden layer
        Dense(d_model)  # Output layer
    ])

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

# Define the Transformer model for classification
def create_transformer_model(input_shape, num_classes):
    model_input = Input(shape=input_shape)
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100)(model_input)
    positional_encoding_layer = positional_encoding(max_length, 100)
    transformer_layer = MultiHeadAttention(num_heads=2, key_dim=100)(embedding_layer, embedding_layer, embedding_layer)
    transformer_layer = Dropout(0.1)(transformer_layer)
    transformer_layer += positional_encoding_layer
    transformer_layer = LayerNormalization(epsilon=1e-6)(transformer_layer)
    transformer_layer = Dense(2048, activation='relu')(transformer_layer)
    transformer_layer = Dense(100)(transformer_layer)
    output_layer = GlobalAveragePooling1D()(transformer_layer)
    output_layer = Dense(num_classes, activation='softmax')(output_layer)
    model = Model(inputs=model_input, outputs=output_layer)
    return model

# Adjust the input shape and number of classes
input_shape = (features_processed.shape[1],)
num_classes = 3  # Assuming 3 classes for password strength
transformer_model = create_transformer_model(input_shape, num_classes)

# Compile the model with appropriate loss function for classification
transformer_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = transformer_model.fit(features_processed, labels, epochs=1, batch_size=64, validation_split=0.2)  

# Save the model
transformer_model.save('transformer_model.h5')

# Save the accuracy
accuracy = history.history['accuracy'][0]  # Get accuracy from training history
with open('transformer_accuracy.txt', 'w') as f:
    f.write(str(accuracy))

# Predict labels using the trained model
predicted_labels = transformer_model.predict(features_processed)
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
