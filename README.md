# Fake News Detection using Passive Aggressive Classifier

This repository contains a Python script for fake news detection using the Passive Aggressive Classifier and TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction. The script uses machine learning techniques to classify news articles as "real" or "fake" based on their content.
# Dataset

The dataset used in this project can be found at fake_or_real_news.csv. It contains labeled news articles, where the "text" column contains the content of each article, and the "label" column indicates whether the article is real or fake.
it is downloaded from Kaggle, link - https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news

# Requirements

    Python 3
    scikit-learn library (includes Passive Aggressive Classifier)
    pandas library
    numpy library
    matplotlib library

You can install the required libraries using pip:

pip install scikit-learn pandas numpy matplotlib

# How it works

    The script loads the dataset from fake_or_real_news.csv.
    The dataset is split into training and testing sets (80% training, 20% testing).
    The content of the news articles (text data) is preprocessed using TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction.
    The Passive Aggressive Classifier model is initialized and trained on the TF-IDF features and their corresponding labels (real or fake).
    The model is then used to predict the labels for the test set.
    The accuracy score of the model is calculated by comparing the predicted labels with the actual labels from the test set.

# How to use

    Clone the repository to your local machine.
    Ensure you have Python 3 and the required libraries installed.
    Run the script:

python fake_news_detection.py

    The script will load the dataset, preprocess the text data using TF-IDF, train the model, and display the accuracy score of the fake news detection.

# Customization

    You can experiment with different machine learning models other than the Passive Aggressive Classifier to see how they perform on the task of fake news detection.
    If you have a different dataset, you can replace the fake_or_real_news.csv file with your dataset, ensuring that it follows a similar format.

# Contributions

Contributions to improve the code and its performance are welcome. If you find any issues or have suggestions for enhancements, please feel free to open an issue or submit a pull request.
