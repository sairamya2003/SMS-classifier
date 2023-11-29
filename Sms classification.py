# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

df = pd.read_csv("train-data.tsv",sep='\t', header= None)
df.columns =['label', 'message']
df.tail()

df.describe()

df.groupby('label').describe().T

# Get all the ham and spam emails
ham_msg = df[df.label =='ham']
spam_msg = df[df.label=='spam']# Create numpy list to visualize using wordcloud
ham_msg_text = " ".join(ham_msg.message.to_numpy().tolist())
spam_msg_text = " ".join(spam_msg.message.to_numpy().tolist())


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
ham_msg_cloud = WordCloud(width =320, height =160, stopwords=STOPWORDS,max_font_size=50, background_color ="tomato", colormap='Blues').generate(ham_msg_text)
plt.figure(figsize=(10,8))
plt.imshow(ham_msg_cloud, interpolation='bilinear')
plt.axis('off') # turn off axis
plt.show()

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
spam_msg_cloud = WordCloud(width =320, height =160, stopwords=STOPWORDS,max_font_size=50, background_color ="white", colormap='autumn').generate(spam_msg_text)
plt.figure(figsize=(10,8))
plt.imshow(spam_msg_cloud, interpolation='bilinear')
plt.axis('off') # turn off axis
plt.show()





import seaborn as sns
plt.figure(figsize=(8,6))
# Percentage of spam messages
(len(spam_msg)/len(ham_msg))*100
plt.xlabel('Bar chart showing imbalanced data')


 #one way to fix it is to downsample the ham msg
ham_msg_df = ham_msg.sample(n = len(spam_msg), random_state = 0)
spam_msg_df = spam_msg
print(ham_msg_df.shape, spam_msg_df.shape)


msg_df = ham_msg_df.append(spam_msg_df).reset_index(drop=True)
plt.figure(figsize=(8,6))
plt.title('Distribution of ham and spam email messages (after downsampling)')


# Get length column for each text
msg_df['text_length'] = msg_df['message'].apply(len) #Calculate average length by label types
labels = msg_df.groupby('label').mean()
labels

df_test = pd.read_csv("valid-data.tsv",sep='\t', header= None)
df_test.columns =['label', 'message']
df_test.tail()

msg_df['msg_type']= msg_df['label'].map({'ham': 0, 'spam': 1})
df_test['msg_type']= df_test['label'].map({'ham': 0, 'spam': 1})
print(msg_df.tail())
print(df_test.tail())

train_label = msg_df['msg_type']
train_msg = msg_df['message']
test_msg = df_test['message']
test_label = df_test['msg_type']
print(test_msg)
print(train_msg)



# Defining pre-processing hyperparameters
max_len = 50
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
vocab_size = 500

vocab_size = 500 # As defined earlier
embeding_dim = 16
drop_value = 0.2 # dropout
n_dense = 24









































