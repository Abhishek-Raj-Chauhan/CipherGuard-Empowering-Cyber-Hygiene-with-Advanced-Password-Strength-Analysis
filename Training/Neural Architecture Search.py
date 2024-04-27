import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Flatten
# Read the File
data = pd.read_csv('training.csv')

# Features which are passwords
features = data['password'].astype(str)

# Labels which are strength of password
labels = data['strength']

# Feature extraction and padding
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(features)
sequences = tokenizer.texts_to_sequences(features)
max_len = max([len(x) for x in sequences])  # Or specify a fixed max length
features_padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# Split data into training and validation sets using padded features
X_train, X_val, y_train, y_val = train_test_split(features_padded, labels, test_size=0.2, random_state=42)
# Convert labels to integers
label_mapping = {'weak': 0, 'medium': 1, 'strong': 2}
y_train = y_train.map(label_mapping)
y_val = y_val.map(label_mapping)

# Define a function to create the NAS model
def create_nas_model(input_shape, num_classes):
    inputs = Input(shape=(input_shape,))  
    x = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=50, input_length=input_shape)(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Flatten the output before passing it to the output layer
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Assuming 3 classes for password strength
    model = Model(inputs, outputs)
    return model


# Create and compile the NAS model
num_classes = len(np.unique(labels))
nas_model = create_nas_model(max_len, num_classes)
nas_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Model output shape:", nas_model.output_shape)
# Train the model
nas_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
nas_model.save('nas_model.h5')
