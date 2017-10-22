# -*- coding: utf-8 -*-
# Classification template

# Importing the libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from ast import literal_eval
import seaborn
from sklearn import metrics 
    
all_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','à','â','œ','ç','è','é','ê','ë','î','ô','ù','û','ä','ö','ß','ü','á','í','ñ','ó','ú','ą','ć','ę','ł','ń','ś','ź','ż','ž','š','č','¿','¡', '\'','ď','ľ','ĺ','ň','ŕ','ť','ý','ï']      

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
    empty_txt = 0
    character_less = 0
    for input_feature, label in zip(input_x, y):
        if isinstance(input_feature, str):
            count_of_letters = get_letters_count(input_feature)
            if (count_of_letters > 10 and count_of_letters < 100):
                cleaned_inputs.append(input_feature) 
                cleaned_labels.append(label)
        else:
            empty_txt+=1

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
        if isinstance(string, str):
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
        if isinstance(string, str):
            for char in string.lower():
                if char not in all_characters and str(char).isdigit() is False and char not in other_chars:
                    other_chars.append(char)
        else:
            if string not in other_chars:
                other_chars.append(string)
    return other_chars

def preprocess_data():
    # Importing the dataset
    dataset_x = pd.read_csv("data/train_set_x.csv")
    X = dataset_x.iloc[:,1]
    dataset_y = pd.read_csv('data/train_set_y.csv')
    Y = dataset_y.iloc[:,1]
    
    cleaned_data = clean_data(X,Y)
    
    cleaned_X = cleaned_data[0]
    cleaned_Y = cleaned_data[1]
    
    extracted_features = extract_features(cleaned_X)
    
    preprocessed_data = {}
    preprocessed_data['X'] = extracted_features
    preprocessed_data['Y'] = cleaned_Y
    
    return preprocessed_data

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
    
    
    # PART 1 -> DATA PREPROCESSING
    data = import_preprocessed_data('cleaned/cleaned_10-100.csv')
    X = data['X']
    Y = data['Y']
    
    # Splitting data to train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.2)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    
    classifier.score(X_test, y_test)
    cm = metrics.confusion_matrix(X_test, y_test)
        
    testset_x = pd.read_csv("data/test_set_x.csv")
    test_X = testset_x.iloc[:,1]
    test_features = extract_features(test_X.tolist())
    test_features = sc.fit_transform(test_features)
    y_test_results = classifier.predict(test_features)
    
    # Export Kaggle test results to submit to competition        
    export_kaggle_results('kaggle/linearLR.csv', 'Id','Category', y_test_results)
    
    

    
    
    
