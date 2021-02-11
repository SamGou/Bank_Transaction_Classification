from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
import gensim
from gensim import models
from gensim.models.ldamodel import LdaModel
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import nltk
nltk.download('punkt')
nltk.download("stopwords")
from nltk.corpus import stopwords

filepath_features = "D:\\FreeAgent\\bank_transaction_features.csv" #Make sure to change to your own path if you want to test run
filepath_labels = "D:\\FreeAgent\\bank_transaction_labels.csv"  #Make sure to change to your own path if you want to test run

features = open(filepath_features,"r")
labels = open(filepath_labels,"r")

F = pd.read_csv(features)
L = pd.read_csv(labels)
df = pd.merge(F,L,on="bank_transaction_id",how="left")
df = df.dropna(axis=0,subset=["bank_transaction_description"])

#Use for plotting
train_set = df[df["bank_transaction_dataset"]=="TRAIN"]
val_set = df[df["bank_transaction_dataset"]=="VAL"]
sets = [train_set,val_set]
categories = train_set["bank_transaction_category"].unique()

stop_words = set(stopwords.words('english'))
# I update stopwords with all months and their variations as well as some very frequent words between all categories
stop_words.update(["jan","Jan","January","feb","Feb","February","mar","Mar","March",
                   "apr","Apr","April","may","May","jun","Jun","June","jul","Jul","July",
                   "aug","Aug","August","sep","Sep","September","oct","Oct","October","nov",
                   "Nov","November","dec","Dec","December","fin","transaction","debit","sundry","direct",
                   "card","payment","cash",":"])
def dates(s):
    # This function creates is used to detect the presence of a date in a description
    counter = 0
    for i in s:
        if any(char.isdigit() for char in i):
            counter += 1
    if counter >= 1:
        return 1
    else:
        return 0

def clean(lst):
    # This function removes any stopwords and digits from the token list
    new_list = [i for i in lst if i.lower() not in stop_words]
    new_list = [i for i in new_list if not any(char.isdigit() for char in i)]
    return new_list

def data_wrangling(sets=sets):
    for _set in sets:

        # Tokenize descriptions for later use
        _set["tokenized"] = _set.apply(lambda row: nltk.word_tokenize(row["bank_transaction_description"]),
                                   axis=1)

        # Create a binary value column to denote the presence of a date in the description
        _set["cont_date"] = _set["tokenized"].apply(dates)

        # Clean the tokens from stopwords and digits
        _set["tokenized_clean"] = _set["tokenized"].apply(clean)
        # Combine the cleaned tokens into full, cleaned descriptions
        _set["clean_desc"] = _set["tokenized_clean"].apply(" ".join)


## EXPLORATORY ANALYSIS
#Is there any difference between the number of transactions and their types, amounts and
#descriptions in each category?

def plotter(df,plot=True):
    size = {} # size of each category
    mostcommon_per_category = {} # top 10 most common words per category
    for i in categories:
        allWords = []
        series = df[df["bank_transaction_category"] == i]["tokenized_clean"]
        for lst in series:
            for word in lst:
                allWords.append(word)
        allWordDist = nltk.FreqDist(w.lower() for w in allWords)
        mostCommon = allWordDist.most_common(10)
        mostcommon_per_category[i] = mostCommon
        size[i] = df[df["bank_transaction_category"] == i].shape[0]


    keys = list(size.keys())
    # Get values in the same order as keys, and parse values
    vals = [int(size[k]) for k in keys]

    # Number of times the transaction category occurs in the dataset
    plt.figure()
    plt.title("Number of times the transaction category occurs in the dataset [{} SET]".format(df["bank_transaction_dataset"].iloc[0]))
    plt.xlabel("Transaction Categories")
    plt.ylabel("Number of occurrences")
    sns.barplot(x=keys, y=vals)
    # Notes: inbalanced data set might have higher accuracy on the categories with more examples, however more false positivies too ?

    #Average transaction amount in each category
    plt.figure()
    plt.title("Plot of the average transaction amount in each category [{} SET]".format(df["bank_transaction_dataset"].iloc[0]))
    sns.barplot(data=df,x="bank_transaction_category",y="bank_transaction_amount",ci="sd")
    plt.xlabel("Transaction Categories")
    plt.ylabel("Average amount")
    # Notes: 2 and 3 are very similar to each other meaning using this feature as a classifier may not be a good idea

    plt.figure()
    plt.title("Plot of the transaction types in each category [{} SET]".format(df["bank_transaction_dataset"].iloc[0]))
    sns.histplot(data=df, x="bank_transaction_category", y="bank_transaction_type",cbar=True)
    plt.xlabel("Transaction Categories")
    plt.ylabel("Transaction Type")
    # Can be used as a feature though not a very good one

    plt.figure()
    plt.title("Plot of the occurrences of dates in each category [{} SET]".format(df["bank_transaction_dataset"].iloc[0]))
    sns.countplot(data=df, x="bank_transaction_category",hue="cont_date")
    plt.xlabel("Transaction Categories")
    plt.ylabel("Number of times a date appeared in the category's descriptions")
    # Not much information can be extracted from this feature as they are all equally likely to either contain or to not contain a date in the description

    dframe = pd.DataFrame(mostcommon_per_category)
    print(dframe)
    plt.show()

##TRAINING
def Vectorizer(type):
    if type == "count":
        # Use words that are at least 3 letters long and combinations of up to 3 words (eg. "transport for london")
        vect = CountVectorizer(ngram_range=(1,3),analyzer="word",token_pattern='(?u)\\b\\w\\w\\w+\\b')
    elif type== "tfidf":
        vect = TfidfVectorizer(ngram_range=(1,3),analyzer="word",token_pattern='(?u)\\b\\w\\w\\w+\\b')
    return vect


def Training(train,corpus,id_map):
    if train or not os.path.isfile("lda.model"):
        # Training will only be performed if model has never been created or if you wish to re-train
        print("TRAINING LDA MODEL.....")
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, id2word=id_map, num_topics=10, passes=50, random_state=34)
        ldamodel.save("lda.model")

#TOPIC PREDICTION
def topic_prediction(train=False):
    global val_set
    global train_set

  # Change to true to enable re-training
    #Use vectoriser for topic modelling with LDAmodel with gensim
    data = train_set["clean_desc"].values
    vect =Vectorizer(type="tfidf")
    X = vect.fit_transform(data)
    corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    id_map = dict((v, k) for k, v in vect.vocabulary_.items())
    Training(train,corpus,id_map)

    val_list = val_set["clean_desc"].values
    X_val = vect.transform(val_list)
    corpus_val = gensim.matutils.Sparse2Corpus(X_val, documents_columns=False)
    doc_topics = list(model.get_document_topics(corpus_val))

    classes = []
    for i in doc_topics:
        import operator
        classes.append(max(i,key=operator.itemgetter(1))[0])

    dataframe = {"classes":classes}
    predictions = pd.DataFrame(dataframe)

    mapping = {0:"INSURANCE",1:"MOTOR_EXPENSES",2:"INSURANCE",3:"TRAVEL",4:"BANK_OR_FINANCE_CHARGES",
               5:"TRAVEL",6:"TRAVEL",7:"ACCOMMODATION_AND_MEALS",8:"ACCOMMODATION_AND_MEALS",9:"TRAVEL"}

    predictions = predictions["classes"].map(mapping)

    true = val_set["bank_transaction_category"].tolist()
    accuracy = accuracy_score(true, predictions)
    return accuracy

#Load model
model = models.LdaModel.load('lda.model')

#RUN
data_wrangling()
# print("ACCURACY:{:.2f} %".format(topic_prediction()*100))

#ANALYSIS
plotter(val_set)