
# Import necessary libraries
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

# Load the dataset from CSV
df = pd.read_csv('fake_or_real_news.csv')

# Check the shape of the dataset (number of rows and columns)
df.shape

# Display the first 10 rows of the dataset
df.head(10)

# Set the 'Unnamed: 0' column as the index of the DataFrame
df.set_index('Unnamed: 0')

# Extract the labels (target variable) from the dataset
y = df.label

# Display the first 5 rows of the 'label' column
y.head(5)

# Drop the 'label' column from the DataFrame to keep only the text data
df = df.drop('label', axis=1)

# Display the first 5 rows of the remaining DataFrame
df.head()

"""# Train Test split"""

# Split the dataset into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=0)

# Check the shape of the training data
x_train.shape

"""# Feature Extraction"""

# Initialize the TF-IDF Vectorizer for feature extraction
count_vectorizer = TfidfVectorizer()

# Fit and transform the training text data to TF-IDF feature vectors
count_train = count_vectorizer.fit_transform(x_train)

# Transform the test text data to TF-IDF feature vectors (using the same vectorizer)
count_test = count_vectorizer.transform(x_test)

# Check the shape of the training data after feature extraction
count_train.shape

# Get the names of the features (words) in the TF-IDF vocabulary
feature_names = count_vectorizer.get_feature_names_out()

# Check the number of features (words) in the TF-IDF vocabulary
(count_vectorizer.get_feature_names_out()).shape

# Create a DataFrame to display the TF-IDF feature vectors
count_df = pd.DataFrame(count_train.toarray(), columns=feature_names)

# Display the first 5 rows of the TF-IDF DataFrame
count_df.head()

"""# Training the Model"""

# Initialize the Passive Aggressive Classifier model
model = PassiveAggressiveClassifier(max_iter=100)

# Train the model using the TF-IDF features and the corresponding labels
model.fit(count_train, y_train)

"""# Prediction and Accuracy score"""

# Make predictions on the test data using the trained model
y_pred = model.predict(count_test)

# Calculate the accuracy score of the model
score = metrics.accuracy_score(y_test, y_pred)

# Display the accuracy score
print('accuracy : %0.3f' % score)
