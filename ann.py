# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 21:17:57 2017

"""

# -*- coding: utf-8 -*-
# Classification template

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from ast import literal_eval
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
    
all_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','à','â','œ','ç','è','é','ê','ë','î','ô','ù','û','ä','ö','ß','ü','á','í','ñ','ó','ú','ą','ć','ę','ł','ń','ś','ź','ż','ž','š','č','¿','¡', '\'','ď','ľ','ĺ','ň','ŕ','ť','ý','ï']      

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 37, kernel_initializer = 'uniform', activation = 'relu', input_dim = 69))
    classifier.add(Dense(units = 37, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

def import_preprocessed_data(file_preprocessed_data):
    preprocessed_data = pd.read_csv(file_preprocessed_data)
    preprocessed_data.X = preprocessed_data.X.apply(literal_eval)
    X = preprocessed_data.iloc[:,0]
    Y = preprocessed_data.iloc[:,1]
    X = X.tolist()
    Y= Y.tolist()
    
    data = {}
    data['X'] = X
    data['Y'] = Y
    
    return data
    
if __name__ == "__main__":
    
    if (len(sys.argv) <3):
        print("Arguments missing")
    else:
        processed_data = sys.argv[1]
        output_model = sys.argv[2]
        # PART 1 -> DATA PREPROCESSING
        data = import_preprocessed_data(processed_data)
        X = data['X']
        Y = data['Y']
        # Convert categorical data 
        Y = keras.utils.to_categorical(Y, 5)
        
        # Splitting data to train set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)
        
        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        # ADAM WITH BINARY CROSS ENTROPY
        classifier = build_classifier()
        classifier.fit(X, Y, batch_size = 32, epochs = 100)
        accuracy = metrics.accuracy_score(y_test, np.argmax(classifier.predict(X_test)))
        print("Accuracy score is " + str(accuracy))
        # Save model to specified file
        classifier.save(output_model)

    

    
    
    
