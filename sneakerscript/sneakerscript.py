import numpy as np
import pandas as pd

#load data
df = pd.read_csv("C:/Users/Student/Desktop/SemesterProject/sneakers_Reviews_Dataset.csv", delimiter = ';')
df.head()

import subprocess

# Install NLTK using pip
subprocess.check_call(['pip', 'install', 'nltk'])

import nltk

nltk.download('punkt') 

##pre processing

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

#convert to lowercase

df.head()

# remove special characters, numbers, and punctuation
df['review_text'] = df['review_text'].apply(lambda text: re.sub(r'[^a-zA-Z\s]', '', text))

df.head()

#tokenization

df['review_text'] = [word_tokenize(text) for text in df['review_text']]

#remove stopwords
stop_words = set(stopwords.words('english'))

df['review_text'] = [[word for word in words if word not in stop_words] for words in df['review_text']]

#stemming (reducing words to their root form)
stemmer = PorterStemmer()
df['review_text'] = [[stemmer.stem(word) for word in words] for words in df['review_text']]

#convert back text
df['review_text'] = [''.join(words) for words in df['review_text']]

#####

#feature extraction

#vectorization (convert text to numerical representation)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['review_text'])

df


#deriving the sentiment frm the rating and text column

def label_sentiment(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

# applying the function
df['sentiment'] = df['rating'].apply(label_sentiment)


#text length

# Install matplotlib using pip
subprocess.check_call(['pip', 'install', 'matplotlib'])

import matplotlib.pyplot as plt

#distribution of sentiment labels
sentiment_counts = df['sentiment'].value_counts()

plt.bar(sentiment_counts.index, sentiment_counts)
plt.title('Distribution of sentiment labels')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

df['text_length'] = df['review_text'].apply(len)

plt.figure(figsize = (10, 6))

for sentiment in df['sentiment'].unique():
    subset = df[df['sentiment'] == sentiment]
    plt.hist(subset['text_length'], bins = 30, alpha = 0.5, label = sentiment)

plt.title('Histogram of Text length in sentiment analysis')
plt.xlabel('Text length')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#analyzing the frequency of words
from wordcloud import WordCloud

text = df['review_text'].str.cat(sep=' ')

# generate a word cloud
wordcloud = WordCloud(width = 800, height = 400, background_color = 'white').generate(text)

#plot the word cloud
plt.figure(figsize = (10, 5))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()

########

# model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# split the data into test/train

del df['review_text']
train = df[df['sentiment', 'timestamp'].notnull().all(axis=1)]
test = df[df['sentiment'].isnull()]
del test['sentiment']

train['sentiment'].value_counts()

train_df = train
test_df = test

X_train, X_test, y_train, y_test = train_test_split(
    train_df.drop(labels=['sentiment', 'timestamp'], axis=1),
    train_df['sentiment'].values,
    test_size=0.2,
    random_state=42
)

# Initialize base models
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

#evaluate performance
report = classification_report(y_test, predictions)
print("Classification Report:")
print(report)



print(f"Accuracy: {accuracy:.4f}")


# Save the vectorizer and model
with open('vectorizer_model.pkl', 'wb') as file:
    pickle.dump({'vectorizer': vectorizer, 'model': model}, file)
with open('sneakerscript.pkl', 'wb') as file:
    pickle.dump(model, file)