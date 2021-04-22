# __Author__ = Marlon Sousa
# __Blog__ = marlonsousa.medium.com

import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup
#from google.drive import drive
import zipfile
import seaborn as sns
import spacy as sp
import string
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers

cols = ["sentiment", "id", "date", "query", "user", "text"]

train_data = pd.read_csv("trainingandtestdata/train.csv", header=None, engine="python", encoding="latin1", names=cols)

test_data = pd.read_csv("trainingandtestdata/test.csv", header=None, engine="python", encoding="latin1", names=cols)
data = train_data
data.drop(["id", "date", "query", "user"], axis=1, inplace=True)
data.head()
X = data.iloc[:, 1].values
y = data.iloc[:, 0].values

from sklearn.model_selection import train_test_split

X, _, y, _ = train_test_split(X, y, test_size=0.85, stratify=y)
unique, counts = np.unique(y, return_counts=True)

def clean_tweets(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    tweet = re.sub(r"@[A-Za-z0-9]+", " ", tweet)
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", " ", tweet)
    tweet = re.sub(r"[^a-zA-Z.!?]", " ", tweet)
    tweet = re.sub(r" +", " ", tweet)
    return tweet


text = "@ddubsbostongirl http://fiap.com.br me of 2course!!!"
text = clean_tweets(text)

nlp = sp.load('en_core_web_sm')

stop_words = sp.lang.en.STOP_WORDS


def clean_tweets2(tweet):
    tweet = tweet.lower()
    document = nlp(tweet)
    
    words = []
    for token in document:
        words.append(token.text)
        
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    words = ' '.join([str(element) for element in words])
        
    return words


data_clean = [clean_tweets2(clean_tweets(tweet)) for tweet in X]

for _ in range(10):
    print(data_clean[random.randint(0, len(data_clean)) - 1])


data_labels = y

data_labels[data_labels == 4] = 1


#np.unique(data_labels)



import tensorflow_datasets as tfds

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(data_clean, target_vocab_size=2**16)

ids = tokenizer.encode("i am happy")


text = tokenizer.decode(ids)

data_inputs = [tokenizer.encode(setence) for setence in data_clean]

for _ in range(10):
    print(data_inputs[random.randint(0, len(data_inputs)) - 1])



max_len = max([len(setence) for setence in data_inputs])

data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs, 
                                                           value=0, 
                                                           padding='post',
                                                           maxlen=max_len)

for _ in range(10):
    print(data_inputs[random.randint(0, len(data_inputs)) - 1])


train_inputs, test_inputs, train_labels, test_labels = train_test_split(data_inputs, data_labels, test_size=0.3, stratify=data_labels)


# Construção do Modelo Base
class DCNN(tf.keras.Model):
    
    def __init__(self, vocab_size, emb_dim=128, nb_filters=50, ffn_units=512, nb_classes=2, dropout_rate=0.1, training=False, name="dcnn"):
        super(DCNN, self).__init__(name=name)
    
        #Camada de Convolução
        self.embedding = layers.Embedding(vocab_size, emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2, padding='same', activation='relu')
        self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3, padding='same', activation='relu')
        self.fourgram = layers.Conv1D(filters=nb_filters, kernel_size=4, padding='same', activation='relu')

        self.pool = layers.GlobalMaxPooling1D()
        
        #Camada Densa
        self.dense_1 = layers.Dense(units=ffn_units, activation='relu')
        self.dropout = layers.Dropout(rate=dropout_rate)
        
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1, activation='sigmoid')
        else:
            self.last_dense = layers.Dense(units=nb_classes, activation='softmax')
            
    
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)
        
        merged = tf.concat([x_1, x_2, x_3], axis=-1) #Batch_size, 3*nb_filters
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output
        
        

vocab_size = tokenizer.vocab_size
emb_dim = 200
nb_filters = 100
ffn_units = 256
nb_classes = len(set(train_labels))
batch_size = 64
dropout_rate = 0.2
nb_epochs = 10

Dcnn = DCNN(vocab_size=vocab_size, emb_dim=emb_dim, nb_filters=nb_filters, ffn_units=ffn_units, 
            nb_classes=nb_classes, dropout_rate=dropout_rate)


if nb_classes == 2:
    Dcnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


checkpoint_path = "./"
ckpt = tf.train.Checkpoint(Dcnn=Dcnn)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.lastest_checkpoint)
    print("Latest Checkpoint Restored")
    

history = Dcnn.fit(train_inputs, train_labels, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_split=0.10)
ckpt_manager.save()


Dcnn.save_weights("Weights.h5")


Dcnn.summary()


results = Dcnn.evaluate(test_inputs, test_labels, batch_size=batch_size)
print(results)

y_pred_test = Dcnn.predict(test_inputs)


y_pred_test = (y_pred_test > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, y_pred_test)
cm

sns.heatmap(cm, annot=True);

history.history.keys()


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss Progress during training validation")
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.legend(["Training Loss", "Validation Loss"]);


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Loss Progress during training validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Training Accuracy", "Validation Accuracy"]);


text = "I love you so much"
text = tokenizer.encode(text)


print(Dcnn(np.array([text]), training=False).numpy())