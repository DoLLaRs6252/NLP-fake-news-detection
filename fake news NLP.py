# %%
#Importing the Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score , recall_score , f1_score,precision_score
from sklearn.metrics  import confusion_matrix,confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer



# %%
#Reading the Dataset
false = pd.read_csv("Fake.csv")
true = pd.read_csv("true.csv")
new = pd.read_csv("csv/fake_or_real_news.csv")
new_true = pd.read_csv("csv/True.csv")
new1 = pd.read_csv("csv/news_articles.csv")
new2 = pd.read_csv("csv/news_dataset.csv")

# %%
new1

# %%
new2

# %%
false.head()

# %%
new.head()

# %%
new_true.head()

# %%
#colum category for fake , true news
true['category'] = 1
false['category'] = 0
true['title'].shape[0]


new['category'] = np.where(new['label']=='FAKE', 0,1)


# %%
new1['category'] = np.where(new1['label']=='Fake', 0,1)

del new1['author']
del new1['published']
del new1['title']
del new1['language']
del new1['site_url']
del new1['main_img_url']
del new1['type']
del new1['title_without_stopwords']
del new1['text']
del new1['hasImage']
new1

# %%
new1 = new1.dropna()
del new1['label']
new1 = new1.rename(columns={"text_without_stopwords": "text"})
new1

# %%
new2['category'] = np.where(new2['label']=='FAKE', 0,1)
del new2['label']
new2

# %%
df3 = pd.concat([new1,new2])
df3

# %%
del new['title']
del new['Unnamed: 0']
del new['label']
new

# %%
df1 = pd.concat([true,false])
#รวมไฟล์2ไฟล์
df1


# %%
#ลบcolumที่ไม่จำเป็น
#df['text'] = df['text'] + " " + df['title']
del df1['title']
del df1['subject']
del df1['date']
df1


# %%
df = pd.concat([df1,new])
df = pd.concat([df,new1])
df = pd.concat([df,new2])
te = df


# %%
df

# %%
#stop = set(stopwords.words('english'))
#punctuation = list(string.punctuation)
#stop.update(punctuation)

# %%
#Data Cleaning def
def wordopt(text):
    text= str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '',text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return(text)

# %%
#uese Data Cleaning
df['text'] = df['text'].apply(wordopt)
print(df['text'])

# %%
#กำหนดตัวทดสอบ กับ เฉลย
x = df['text']
y = df['category']

# %%
df

# %%
#text to Matrix and split data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
vec_train = CountVectorizer().fit(x_train)
X_vec_train = vec_train.transform(x_train)
X_vec_test = vec_train.transform(x_test)

# %%
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_vec_train)
X_test_tfidf = tfidf_transformer.fit_transform(X_vec_test)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
LG = LogisticRegression(C=2,max_iter=5000)
LG.fit(X_vec_train, y_train)
LGpredicted_value = LG.predict(X_vec_test)
print("Accuracy score: ",accuracy_score(y_test, LGpredicted_value))


# %%
print("Accuracy score: ",accuracy_score(y_test, LGpredicted_value))
print("precision score: ",precision_score(y_test, LGpredicted_value))
print("f1 score: ",f1_score(y_test, LGpredicted_value))
print("Recall score: ",recall_score(y_test, LGpredicted_value))
cm = confusion_matrix(y_test, LGpredicted_value)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=LG.classes_)
disp.plot()
plt.show()



# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
Rd1 = RandomForestClassifier(max_depth=2, random_state=0)
Rd1.fit(X_vec_train, y_train)
Rd1predicted_value = Rd1.predict(X_vec_test)
print("Accuracy score: ",accuracy_score(y_test, Rd1predicted_value))
print("precision score: ",precision_score(y_test, Rd1predicted_value))
print("f1 score: ",f1_score(y_test, Rd1predicted_value))
print("Recall score: ",recall_score(y_test, Rd1predicted_value))
cm = confusion_matrix(y_test, Rd1predicted_value)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Rd1.classes_)
disp.plot()
plt.show()


# %%
print(len(x))
print(len(y))

# %%
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,max_depth=1, random_state=0)
GB.fit(X_vec_train, y_train)
GBpredicted_value = GB.predict(X_vec_test)
y_pred = GB.predict(X_vec_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy score: ",accuracy_score(y_test, y_pred))
print("precision score: ",precision_score(y_test, y_pred))
print("f1 score: ",f1_score(y_test, y_pred))
print("Recall score: ",recall_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=GB.classes_)
disp.plot()
plt.show()


# %%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

# transforming
tfidf_train = vectorizer.fit_transform(x_train)
tfidf_test = vectorizer.transform(x_test)

# %%
LGtf = LogisticRegression(C=2,max_iter=5000)
LGtf.fit(tfidf_train, y_train)
LGtfpredicted_value = LGtf.predict(tfidf_test)
print("Accuracy score: ",accuracy_score(y_test, LGtfpredicted_value))
print("precision score: ",precision_score(y_test, LGtfpredicted_value))
print("f1 score: ",f1_score(y_test, LGtfpredicted_value))
print("Recall score: ",recall_score(y_test, LGtfpredicted_value))
cm = confusion_matrix(y_test, LGtfpredicted_value)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=LGtf.classes_)
disp.plot()
plt.show()


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
Rd = RandomForestClassifier(max_depth=2, random_state=0)
Rd.fit(tfidf_train, y_train)
Rdpredicted_value = Rd.predict(tfidf_test)
print("Accuracy score: ",accuracy_score(y_test, Rdpredicted_value))
print("precision score: ",precision_score(y_test, Rdpredicted_value))
print("f1 score: ",f1_score(y_test, Rdpredicted_value))
print("Recall score: ",recall_score(y_test, Rdpredicted_value))
cm = confusion_matrix(y_test, Rdpredicted_value)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Rd.classes_)
disp.plot()
plt.show()


# %%
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')  # Download NLTK tokenizer data if not already done


# Tokenize the sentences
te['token'] = [word_tokenize(sentence.lower()) for sentence in te['text']]

# Train the Word2Vec model
model = Word2Vec(sentences=te['token'], vector_size=100, window=5, min_count=1, sg=1)

# Save the model
model.save("word2vec_example.model")

# Load the model
# model = Word2Vec.load("word2vec_example.model")

# Find similar words
similar_words = model.wv.most_similar('a')
print("Words similar :", similar_words)


# %%
embedded_data = []
for text in te['token']:
    
    
    # Initialize an empty list to store word embeddings
    embeddings = []
    
    for word in text:
        if word in model.wv:
            embedding = model.wv[word]
            embeddings.append(embedding)
    
    # Calculate the sentence embedding by averaging word vectors
    if embeddings:
        sentence_embedding = np.mean(embeddings, axis=0)
        embedded_data.append(sentence_embedding)
    else:
        # Handle out-of-vocabulary words or empty text as needed
        embedded_data.append(None)

# Add the embedded data as a new column in the DataFrame
te['embed'] = embedded_data

# %%
print("Number of samples in y:", len(y))
te = te.dropna()
te

# %%
X_train, X_test, y_train, y_test = train_test_split(te['embed'].to_list(), te['category'], test_size=0.3,random_state=42)

l = LogisticRegression(C=25,max_iter=5000)
l.fit(X_train, y_train)

# Evaluate the model
y_pred = l.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy score: ",accuracy_score(y_test, y_pred))
print("precision score: ",precision_score(y_test, y_pred))
print("f1 score: ",f1_score(y_test, y_pred))
print("Recall score: ",recall_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=l.classes_)
disp.plot()
plt.show()





# %% [markdown]
# 

# %%
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,max_depth=1, random_state=0)
GB.fit(X_train, y_train)
GBpredicted_value = GB.predict(X_test)
y_pred = GB.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy score: ",accuracy_score(y_test, y_pred))
print("precision score: ",precision_score(y_test, y_pred))
print("f1 score: ",f1_score(y_test, y_pred))
print("Recall score: ",recall_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=GB.classes_)
disp.plot()
plt.show()


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

Rd = RandomForestClassifier(max_depth=60, random_state=0)
Rd.fit(X_train, y_train)
Rdpredicted_value = Rd.predict(X_test)
print("Accuracy score: ",accuracy_score(y_test, Rdpredicted_value))
print("precision score: ",precision_score(y_test, Rdpredicted_value))
print("f1 score: ",f1_score(y_test, Rdpredicted_value))
print("Recall score: ",recall_score(y_test, Rdpredicted_value))
cm = confusion_matrix(y_test, Rdpredicted_value)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=Rd.classes_)
disp.plot()
plt.show()



