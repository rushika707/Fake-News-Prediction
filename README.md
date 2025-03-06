# Fake-News-Prediction

## Overview
The Fake News Prediction System is a machine learning-based application designed to classify news articles as **real** or **fake**. It leverages **Natural Language Processing (NLP)** techniques and **machine learning algorithms** to analyze textual data and determine its authenticity.

## Features
- **Preprocessing**: Cleans and prepares news text by removing stopwords, stemming, and vectorizing.
- **TF-IDF Vectorization**: Converts text into numerical format for model processing.
- **Machine Learning Model**: Uses **Logistic Regression** for classification.
- **Evaluation Metrics**: Measures model performance using **accuracy, precision, recall, and F1-score**.
- **Interactive Interface**: Accepts user-inputted news text for real-time prediction.

## Tech Stack
- **Programming Language**: Python
- **Libraries**: `numpy`, `pandas`, `sklearn`, `nltk`, `matplotlib`, `seaborn`
- **Machine Learning Algorithm**: Logistic Regression


## Model Training
1. **Load dataset**: Pre-processed fake news dataset.
2. **Text cleaning**: Tokenization, stopword removal, and stemming.
3. **Feature extraction**: TF-IDF vectorization.
4. **Model training**: Logistic Regression with train-test split.
5. **Evaluation**: Performance metrics calculated on the test dataset.

## Future Enhancements
- Integration of **Deep Learning (LSTM, Transformers)** models for improved accuracy.
- Deployment as a **web application** using Flask/Django.
- Expansion to support **multilingual news classification**.
- **Explainability features** to highlight reasons behind predictions.

