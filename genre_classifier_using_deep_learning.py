
# Commented out IPython magic to ensure Python compatibility.
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import csv# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler#Keras
import math
import keras
from keras import models
from keras import layers

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
files.upload()

!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download andradaolteanu/gtzan-dataset-music-genre-classification -p music_dataset

!unzip /content/music_dataset/gtzan-dataset-music-genre-classification.zip -d /content/music_dataset/gtzan

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

df=pd.read_csv('/content/data.csv')
df.head()

!gsutil cp -r  /content/music_dataset/gtzan /content/drive/MyDrive

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    
    for filename in os.listdir(f'/content/drive/MyDrive/gtzan/Data/genres_original/{g}'):
        songname = f'/content/drive/MyDrive/gtzan/Data/genres_original/{g}/{filename}'
        
        print(filename)
        if(filename=='.ipynb_checkpoints'):
          continue
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

data1 = pd.read_csv('/content/drive/MyDrive/data.csv')
data1 = data1.drop_duplicates(keep='last')
data1[50:100]

!gsutil cp   /content/data.csv /content/drive/MyDrive

data1.shape

data1 = data1.drop(['filename'],axis=1)
data1.head()

genre_list = data1.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
print(y)
len(y)

scaler = StandardScaler()
X = scaler.fit_transform(np.array(data1.iloc[:, :-1], dtype = float))
X.shape
print(scaler)
import pickle
filename = 'scaler.pkl'
pickle.dump(scaler, open(filename, 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = models.Sequential()
X_train[100]

a=[-0.14492835,  1.00400067 , 0.43310275 , 0.54454623,  0.75775479,  0.40733995,
  0.87956002 ,-0.36890081 , 0.34177886  ,0.29173232 ,-0.36187661  ,1.10997991,
 -1.89878167 , 1.11036123 ,-1.38510493  ,0.57681272 ,-1.13252072  ,1.09590156,
  0.13179925 , 1.05871994 ,-0.12250645  ,0.86477539 ,-0.59280286  ,1.38374326,
 -0.36733292 ,-0.23117629]
a = np.array([song])
for i in range(0,10):
   if(X_test[i].all==a.all):
     print(i)
X_test[1]     
y_test[1]

model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=64)

test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_loss)
print('test_acc: ',math.ceil(test_acc*100),'%')

a=np.array([[0.260574 	,0.051349 	,1132.340629 	,1582.492047 	,2065.479177 	,0.048314 	,-314.582794 	,139.926743 	,11.865850 	,30.113317 	,-7.549764 	,14.914179 ,	-2.699563 ,	6.707931 ,	-6.391602 	,2.819350 	,-11.715414 ,	2.077480 	,-11.442501 	,-9.504389 	,-15.185495 ,	-9.355572 ,	-5.672771, 	1.564773 	,0.890936 ,-7.960377]])
prediction=loaded_model.predict(a)
print(prediction)

predictio=model.predict(np.array([X_test[0]]))
for i in predictio[0]:
  print(i)
#np.argmax(prediction)#this is being predicted for a single song,thats what we will do when upload...

from keras.models import load_model
model.save("network.h5")

!gsutil cp   /content/network.h5 /content/drive/MyDrive

loaded_model = load_model("/content/drive/MyDrive/network.h5")
loss, accuracy = loaded_model.evaluate(X_test, y_test)

# predictions = loaded_model.predict(X_test)
# for i in range(0,199):
#   print(np.argmax(predictions[i]))
print(X_test[1])
print(datax)
predictions = loaded_model.predict(X_train)

for i in range(0,100):
  
  print(np.argmax(predictions[i]),y_train[i])

