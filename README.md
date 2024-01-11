# Hate Speech Classification
# Overview

This repository contains code and resources for a machine learning project focused on classifying comments as abusive or non-abusive using different techniques. The project is divided into three parts: TF-IDF + KNN classification, LSTM-based classification, and Multilingual language model classification using mBERT and MuRIL.
Dataset

The dataset consists of around 20,000 comments in the Hindi language, labeled as either abusive (0) or non-abusive (1). The dataset is provided in the Google Drive link. The comments have been preprocessed, and each data point includes the comment text and its corresponding label.
Data Fields

    Comment Text: The text content of the comment.
    Label: A binary label indicating whether the comment is abusive (0) or non-abusive (1).

# TF-IDF + KNN Classification 
Steps:

    Data Preprocessing:
        Remove unnecessary symbols, stop words, etc.
        Emojis are retained for classification.

    TF-IDF Feature Extraction:
        Utilize TF-IDF for feature extraction to create a feature matrix.

    KNN Classification:
        Determine the optimal 'K' for the KNN classifier using a validation set.
        Explore other parameters for optimal model performance.

    Classification Process:
        For each example in the training data, calculate the distance to the query example using KNN.
        Return the mode of the K labels for classification.

# LSTM-based Classification
Steps:

    Data Preprocessing:
        Similar to Task 1.

    Vocabulary Creation:
        Create the vocabulary for the LSTM model.

    Sequence Processing:
        Define the maximum sequence length.
        Tokenize and pad/truncate sentences accordingly.

    Data Splitting:
        Split the data into training and validation sets (80-20 ratio).

    Model Definition:
        Define the model architecture, including embedding layer, LSTM layer, classification layer, dropouts, and activation functions.

    Training:
        Train the model on the training data.
        Monitor validation set performance at the end of each epoch.
        Implement early stopping if performance doesn't improve over a specified number of iterations.

    Testing:
        Evaluate the model on the test set (hidden for evaluation).
