#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:12:52 2017

@author: orujahmadov
"""

# Importing the libraries
import pandas as pd
import csv
import sys

    
all_characters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','à','â','œ','ç','è','é','ê','ë','î','ô','ù','û','ä','ö','ß','ü','á','í','ñ','ó','ú','ą','ć','ę','ł','ń','ś','ź','ż','ž','š','č','¿','¡', '\'','ď','ľ','ĺ','ň','ŕ','ť','ý','ï']      

# Counting the total number of occurance of characters from predefine array above
# string: String to count characters of
# returns the total number of characters
def get_letters_count(string):
    counter = 0
    for character in all_characters:
        counter+=string.lower().count(character)
        
    return counter
        
# Filtering data by applying lower and upper boundary to characters counts
# input_x: train set X values
# y : train set labels
# returns cleaned data
def clean_data(input_x, y):
    cleaned_inputs = []
    cleaned_labels = []
    cleaned_data = []
    for input_feature, label in zip(input_x, y):
        if isinstance(input_feature, str):
            count_of_letters = get_letters_count(input_feature)
            if (count_of_letters > 10 and count_of_letters < 100):
                cleaned_inputs.append(input_feature) 
                cleaned_labels.append(label)

    cleaned_data.append(cleaned_inputs)
    cleaned_data.append(cleaned_labels)
    
    return cleaned_data
    
# Counts the frequency of each character from predefined array
# input_string: to count character frequency of
# returns character frequency array
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

# Helper function to export cleaned data to indicated file
def export_cleaned_data(file_name, clean_x, clean_y):
    with open(file_name, 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        index = 0
        filewriter.writerow(['X','Y'])
        for x,y in zip(clean_x, clean_y):
            filewriter.writerow([x,y])
            index+=1

def preprocess_data(file_x, file_y):
    # Importing the dataset
    dataset_x = pd.read_csv(file_x)
    X = dataset_x.iloc[:,1]
    dataset_y = pd.read_csv(file_y)
    Y = dataset_y.iloc[:,1]
    
    cleaned_data = clean_data(X,Y)
    
    cleaned_X = cleaned_data[0]
    cleaned_Y = cleaned_data[1]
    
    extracted_features = extract_features(cleaned_X)
    
    preprocessed_data = {}
    preprocessed_data['X'] = extracted_features
    preprocessed_data['Y'] = cleaned_Y
    
    return preprocessed_data

if __name__ == "__main__":
    
    if (len(sys.argv) <4):
        print("Arguments missing")
    else:
        file_x = sys.argv[1]
        file_y = sys.argv[2]
        output_file = sys.argv[3]
        data = preprocess_data(file_x, file_y)
        export_cleaned_data(output_file, data['X'], data['Y'])
        
        