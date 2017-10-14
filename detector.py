# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from nltk import ngrams


all_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# PART 1 DATA PREPROCESSING

# Importing the dataset
dataset_x = pd.read_csv("train_set_x.csv")
X = dataset_x.iloc[:,1]
dataset_y = pd.read_csv('train_set_y.csv')
Y = dataset_y.iloc[:,1]

# CLEANING

# Feature Extraction
def get_letters_frequency(input_string):
    letters_frequency = []
    for character in all_characters:
        letters_frequency.append(input_string.lower().count(character))
    return letters_frequency

def extract_features(input_data):
    features_array = []
    for string in input_data:
        if isinstance(string, basestring):
            features_array.append(get_letters_frequency(string))
        else:
            features_array.append(get_letters_frequency(" "))
    return features_array

input_features = extract_features(X.tolist())

X_train, X_test, y_train, y_test = train_test_split(input_features, Y.tolist(), random_state=0)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print(classifier.score(X_test, y_test))
