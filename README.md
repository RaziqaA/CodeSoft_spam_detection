
# SMS Spam Detection 

## Introduction
This project focuses on detecting spam messages from a given dataset using Natural Language Processing (NLP) and machine learning techniques. The SMS data is pre-processed, tokenized, and transformed into a format suitable for building a classification model that identifies whether a message is spam or not.

## Dataset
The dataset consists of **5,572** SMS messages, categorized into two classes:
- **Ham**: Non-spam messages.
- **Spam**: Unwanted or promotional messages.

After data cleaning, the final dataset consists of **5,169** messages due to the removal of duplicates and missing data.

### Dataset Columns:
- **target**: Label indicating whether a message is spam (1) or ham (0).
- **Text**: The actual content of the SMS message.
- **num_characters**: Number of characters in the message.
- **num_words**: Number of words in the message.
- **num_sentences**: Number of sentences in the message.

## Project Structure

1. **Data Cleaning**:
   - Removed irrelevant columns.
   - Dropped duplicate and null values.
   - Converted text to lowercase.
   - Tokenized the messages into words.
   - Removed stopwords and punctuation.
   - Applied stemming to reduce words to their root form.

2. **Exploratory Data Analysis (EDA)**:
   - Generated word clouds for spam and ham messages.
   - Visualized word frequency using bar charts.
   - Generated histograms and pair plots to analyze distributions and relationships.
   - Correlation analysis between message length, word count, sentence count, and target variable.

3. **Feature Engineering**:
   - Generated new features: `num_characters`, `num_words`, and `num_sentences`.
   - Applied TF-IDF vectorization to transform the text data into numerical form.

4. **Model Building**:
   - Used three Naive Bayes classifiers: **GaussianNB**, **MultinomialNB**, and **BernoulliNB**.
   - Split the dataset into training and testing sets using an 80/20 split.

5. **Model Evaluation**:
   - Evaluated the performance using:
     - **Accuracy Score**
     - **Confusion Matrix**
     - **Precision Score**

6. **Improvements**:
   - Further improvements could involve tuning hyperparameters or trying other classification algorithms like Support Vector Machines (SVM) or Random Forest.

## Usage

### 1. Data Preprocessing
- The dataset is cleaned and transformed using text preprocessing techniques such as tokenization, stemming, and removal of stop words and punctuation.

### 2. EDA and Visualization
- Visualize the text data distributions and word clouds for spam and ham messages.

### 3. Model Building
- Build a machine learning model using Naive Bayes (MultinomialNB) on TF-IDF transformed text data.

### 4. Evaluation
- Evaluate model performance on the test set using accuracy and precision scores.

### Example Code to Run

```python
# Data Preprocessing
df['transformed_text'] = df['Text'].apply(transform_text)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Model Building and Evaluation
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

# Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
```

## License
This project is licensed under the MIT License.

