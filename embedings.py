from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
#from tensorflow.keras.processing.sequence import pad_sequences
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential

### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

## Define the vocabulary size
voc_size=10000

### One Hot Representation
one_hot_repr = [one_hot(words,voc_size)for words in sent]


## word Embedding Representation

sent_Length = 8
embedded_docs = pad_sequences(one_hot_repr,padding="pre",maxlen=sent_Length)


## feature representation
dim=15

model = Sequential()
model.add(Embedding(voc_size,dim,input_length=sent_Length))
model.compile("adam","mse")

model.summary()
print(model.predict(embedded_docs[0]))
