Movie Genre Classification
This project was developed as part of my internship at CodSoft. The objective is to build a machine learning model that classifies movies into genres based on their plot summaries using Natural Language Processing (NLP) techniques.

Project Overview:

Built a complete text classification pipeline from raw text to genre prediction.

Used TF-IDF vectorization to transform text into features.

Applied Logistic Regression for model training and prediction.

Evaluated the model using accuracy, precision, recall, and F1-score.

Performed Exploratory Data Analysis (EDA) on genre distribution and description length.

Dataset Details:

train_data.txt – Contains movie ID, title, genre, and description.

test_data.txt – Contains movie ID, title, and description.

test_data_solution.txt – Contains the actual genres for test data.

description.txt – Reference file explaining the format.

Steps Performed:

Loaded and explored the dataset.

Preprocessed text: lowercased, removed punctuation, removed stopwords, and applied stemming.

Extracted features using TF-IDF.

Trained a Logistic Regression model on the processed text.

Predicted genres for the test data.

Evaluated the model using the provided solution file.

Conducted EDA to understand the distribution and characteristics of the data.

Technologies and Libraries Used:

Python

Pandas

Scikit-learn

NLTK

Matplotlib

Seaborn

Learning Outcomes:

Gained hands-on experience in real-world text classification.

Learned the full pipeline: preprocessing, feature engineering, model building, and evaluation.

Strengthened my understanding of TF-IDF, Logistic Regression, and NLP workflows.

Future Improvements:

Try other classifiers such as Naive Bayes or Support Vector Machines.

Apply hyperparameter tuning for better performance.

Experiment with word embeddings.

Handle multi-label classification if applicable.

