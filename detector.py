# -*- coding: utf-8 -*-
# Classification template

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
import sys

all_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','à','â','œ','ç','è','é','ê','ë','î','ô','ù','û','ä','ö','ß','ü','á','í','ñ','ó','ú','ą','ć','ę','ł','ń','ś','ź','ż','ž','š','č','¿','¡', '\'','ď','ľ','ĺ','ň','ŕ','ť','ý','ï']      

# PART 1 -> DATA PREPROCESSING

# Importing the dataset
dataset_x = pd.read_csv("data/train_set_x.csv")
X = dataset_x.iloc[:,1]
dataset_y = pd.read_csv('data/train_set_y.csv')
Y = dataset_y.iloc[:,1]

def get_letters_count(string):
    counter = 0
    for character in all_characters:
        counter+=string.lower().count(character)
        
    return counter
        
# CLEANING
def clean_data(input_x, y):
    cleaned_inputs = []
    cleaned_labels = []
    cleaned_data = []
    for input_feature, label in zip(input_x, y):
        if isinstance(input_feature, basestring):
            if (get_letters_count(input_feature) > 40 and get_letters_count(input_feature) < 100):
                cleaned_inputs.append(input_feature) 
                cleaned_labels.append(label)
    
    cleaned_data.append(cleaned_inputs)
    cleaned_data.append(cleaned_labels)
    
    return cleaned_data
    
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

def export_kaggle_results(file_name, header1_name, header2_name, results):
    with open(file_name, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([header1_name, header2_name])
        index = 0
        for result in results:
            filewriter.writerow([index,result])
            index+=1

def export_cleaned_data(file_name, clean_x, clean_y):
    with open(file_name, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        index = 0
        for x,y in zip(clean_x, clean_y):
            filewriter.writerow([x,y])
            index+=1

def helper_function(input_x):
    other_chars = []
    for string in input_x:
        if isinstance(string, basestring):
            for char in string.lower():
                if char not in all_characters and str(char).isdigit() is False and char not in other_chars:
                    other_chars.append(char)
        else:
            if string not in other_chars:
                other_chars.append(string)
    return other_chars


if __name__ == "__main__":
    
    cleaned_data = clean_data(X,Y)
    
    cleaned_X = cleaned_data[0]
    cleaned_Y = cleaned_data[1]
    
    input_features = extract_features(cleaned_X)
    
    X_train, X_test, y_train, y_test = train_test_split(input_features, cleaned_Y, random_state=0, test_size=0.1)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    
    from sklearn.ensemble import RandomForestClassifier
    # PART 2 -> TRAINING
    classifier = RandomForestClassifier(max_depth=46)
    classifier.fit(X_train,y_train)
    
    print("Clasifier score is " + str(classifier.score(X_test, y_test)))

    testset_x = pd.read_csv("data/test_set_x.csv")
    test_X = testset_x.iloc[:,1]
    test_features = extract_features(test_X.tolist())
    test_features = sc.fit_transform(test_features)
    y_test_results = classifier.predict(test_features)
        
    export_kaggle_results('kaggle_forest92.csv', 'Id','Category', y_test_results)    
    
    
