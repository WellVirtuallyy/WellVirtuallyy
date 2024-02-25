import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Initialize NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Initialize variables
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Load intents data
try:
    with open('intents.json', 'r') as file:
        intents = json.load(file)
except FileNotFoundError:
    print("File 'intents.json' not found.")
    exit()
except json.JSONDecodeError:
    print("Error decoding JSON from 'intents.json'.")
    exit()

# Process intents data
try:
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            # Add documents in the corpus
            documents.append((w, intent['tag']))
            # Add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
except KeyError:
    print("Key 'intents' not found in the JSON data.")
    exit()

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Save words and classes to pickle files
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create training data
training = []

# Create an empty array for our output
output_empty = [0] * len(classes)

# Training set: bag of words for each sentence
for doc in documents:
    # Initialize our bag of words
    bag = []
    # List of tokenized words for the pattern
    pattern_words = doc[0]
    # Lemmatize each word - create base word, in an attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create our bag of words array with 1 if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle features and turn into np.array
random.shuffle(training)
training = np.array(training)

# Create train and test lists: X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("Training data created")

# Create model - 3 layers
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)
print("Model created successfully.")
