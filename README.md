# Human Activity Recognition (HAR) Assignment
Overview
This repository contains the implementation and analysis of Human Activity Recognition (HAR) using accelerometer data from the UCI-HAR dataset. The assignment covers:

# Exploratory Data Analysis (EDA)
Decision Tree modeling
Prompt Engineering with LLMs
Data Collection in real-world settings
Dataset
The UCI-HAR dataset is used for classifying human activities based on raw accelerometer data. The dataset is preprocessed using provided scripts:

CombineScript.py
MakeDataset.py
These scripts generate train, test, and validation splits.

# Tasks Completed
# Task 1: Exploratory Data Analysis (EDA)
  Plotted waveforms for different activity classes.
  Analyzed acceleration values to differentiate between static and dynamic activities.
  Applied PCA and TSFEL for feature extraction and visualization.
  Compared different feature extraction methods.
  Computed correlation matrix to identify redundant features.
# Task 2: Decision Trees for HAR
  Implemented Decision Tree models using:
  Raw accelerometer data
  TSFEL features
  Dataset-provided features
  Evaluated models using accuracy, precision, recall, and confusion matrix.
  Trained Decision Trees with varying depths (2-8) and analyzed performance.
  Investigated poor-performing participants/activities.
# Task 3: Prompt Engineering for LLMs
  Demonstrated Zero-Shot and Few-Shot Learning for activity classification.
  Compared Few-Shot Learning with Decision Trees quantitatively.
  Analyzed limitations of Zero-Shot and Few-Shot Learning.
  Tested model behavior with unseen activities and random data.
# Task 4: Data Collection in the Wild
  Collected real-world accelerometer data using a mobile app.
  Preprocessed data and compared model performance on real-world data.
  Applied Few-Shot Learning to classify collected data.
  Reported results and compared with UCI-HAR dataset results.
  Decision Tree Implementation
  Implemented Decision Trees from scratch in tree/base.py.
  Supported different cases:
  Discrete and real features
  Discrete and real outputs
  Used Information Gain (Entropy/Gini Index) for splitting.
  Validated implementation using usage.py.
# Note
  We have done it in group of four
