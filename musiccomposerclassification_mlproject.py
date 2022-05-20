
import librosa
import math
import os

from google.colab import drive
drive.mount('/content/gdrive')

SAMPLE_RATE = 22050 # STANDARD CONSUMER GRADE SAMPLE RATE

def save_mfcc(dataset_path,
              json_path,
              ignore = [],
              n_mfcc = 13,
              n_fft = 2048,
              hop_length = 512,
              num_segments = 10):
    """
    dataset_path: location of the dataset typically folder
    json_path: location where to save output json file
    n_mfcc: number of mfcc coffienant
    n_fft: number of fourier transform [ HAVE TO CHECK ]
    hop_length: length of single segment
    num_segment: total number of segment
    """

    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
    }
    """
    mapping : Name of the composers
            : ["Morzart", "Beethovan", "Bach", ...]
    mfcc    : mfcc coff value in the 13(default) columned array,
            : [[...], [...], [...]]
    labels  : index of mapped composers
    """

    # loop in the dataset directory expect ignored root directory
    for idx, (root, dirs, files) in enumerate(os.walk(dataset_path)):

        if not root in ignore:
            composer = root.split("/")[-1]
            data["mapping"].append(composer)
            print(composer)

            for file in files:
                print(f"processing: {idx}. {composer} -> {file}")

                file_path = os.path.join(root, file)
                
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                duration = librosa.get_duration(signal, sr=SAMPLE_RATE)
                
                # full SAMPLE RATE of the track
                SAMPLES_PER_TRACK = SAMPLE_RATE * duration
                # number of segments in the track
                num_samples_per_segment = SAMPLES_PER_TRACK // num_segments

                # expected mfcc size for a segment
                expected_num_mfcc_vector_per_segment = math.ceil(num_samples_per_segment / hop_length)
                """
                This is required to eliminate those last segment of the
                track which are sorter then the required length
                """

                for s in range(num_segments):
                    start_sample = int(num_samples_per_segment * s)
                    finish_sample = int(start_sample + num_samples_per_segment)

                    # getting mfcc vector
                    mfcc = librosa.feature.mfcc(
                        signal[start_sample:finish_sample],
                        sr=sr,
                        n_fft=n_fft,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length
                    )

                    mfcc = mfcc.T

                    if len(mfcc) == expected_num_mfcc_vector_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append((idx - 1))

                        print(f"{file}, segment: {s}")
    return data

path = "/content/gdrive/MyDrive/MLProject/Dataset/"
ignore = ['/content/gdrive/MyDrive/MLProject/Dataset/', '/content/gdrive/MyDrive/MLProject/Dataset/zip', '/content/gdrive/MyDrive/MLProject/Dataset/zip/.ipynb_checkpoints']
json_path = '.'

data = save_mfcc(path, json_path, ignore)

import json
with open("/content/gdrive/MyDrive/MLProject/Dataset/mfcc.json", "w") as file:
    json.dump(data, file, indent=4)

"""### Converting JSON to CSV
This is for simplicity
"""

import csv
import json

with open("/content/gdrive/MyDrive/MLProject/Dataset/mfcc.json", 'r') as file:
    json_data = json.load(file)

f_name = "/content/gdrive/MyDrive/MLProject/Dataset/mfcc.csv"

with open(f_name, 'w') as file:
    csv_file = csv.writer(file)
    csv_file.writerow(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "Composer"])

    for mfccs, label in zip(json_data['mfcc'], json_data['labels']):
        for idx, mfcc in enumerate(mfccs):
            print(f"---> Processing {idx} out of {len(mfccs)} mfccs for label {label}")
            csv_file.writerow([*mfcc, label])

"""### Spliting into Train and test"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Model_DIR = "./gdrive/MyDrive/MLProject/Dataset/Models/"

# Shuffled dataset
dataset = pd.read_csv("./gdrive/MyDrive/MLProject/Dataset/mfcc.csv")
dataset=dataset.sample(frac=1)
dataset.head()

print("Number of rows: %d" % len(dataset))

X = dataset.drop([ "Composer"], axis=1)
Y = dataset.Composer

from sklearn.model_selection import train_test_split

# what is the meaning of life the universe and everything?
# its 42 
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, 
    random_state=42, 
    train_size=0.8 # 80% train dataset
)

print(f"x_train.shape: {x_train.shape},\t x_test.shape: {x_test.shape}")
print(f"y_train.shape: {y_train.shape},\t y_test.shape: {y_test.shape}")

# 3d plot of the first 300 values in dataset using first 3 mfcc coff

# Shuffling dataset
dummy_data = dataset.sample(frac=1)[:300]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for s in dummy_data.Composer.unique():
    ax.scatter(
        dummy_data['1'] [ dummy_data.Composer == s ],
        dummy_data['2'] [ dummy_data.Composer == s ],
        dummy_data['3'] [ dummy_data.Composer == s ],
        label = s
    )

ax.set_xlabel("Coff 1")
ax.set_ylabel("Coff 2")
ax.set_zlabel("Coff 3")
    
ax.legend()
fig.savefig('plot.png')

"""# Using SKLearn"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
"""
As dataset is very large, so we can standardize the
data set with StandardScale

z = (x - u) / s
z = after standardising
x = value
u = mean of the dataset
s = standard diviation of the dataset
"""
from sklearn.preprocessing import StandardScaler

# ovr is one vs rest
LR_pipeline = make_pipeline(StandardScaler(), LogisticRegression(multi_class="ovr"))
LR_pipeline.fit(x_train, y_train)

# Predictions
LR_y_pred = LR_pipeline.predict(x_test)
LR_y_train_pred = LR_pipeline.predict(x_train)
LR_y_train_pred = LR_pipeline.predict(a)
LR_y_train_pred

# calculating other require values
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

LR_test_r2 = r2_score(y_test, LR_y_pred)
LR_train_r2 = r2_score(y_train, LR_y_train_pred)

LR_test_mse = mean_squared_error(y_test, LR_y_pred)
LR_train_mse = mean_squared_error(y_train, LR_y_train_pred)

LR_test_accuracy = accuracy_score(y_test, LR_y_pred)
LR_train_accuracy = accuracy_score(y_train, LR_y_train_pred)

print("Test r2_score %.2f" % LR_test_r2)
print("Train r2_score %.2f" % LR_train_r2)

print("\nTest mean squared error %.2f" % LR_test_mse)
print("Train mean squared error %.2f" % LR_train_mse)

print("\nTest accuracy score %.2f" % LR_test_accuracy)
print("Train accuracy score %.2f" % LR_train_accuracy)

print("For test dataset")
print(classification_report(y_test, LR_y_pred))

print("\nFor train dataset")
print(classification_report(y_train, LR_y_train_pred))

conf_mat = confusion_matrix(y_test, LR_y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=dataset['Composer'].unique(), 
            yticklabels=dataset['Composer'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - Logistic Regression\n", size=16);

# saving models
import pickle
DIR = "./gdrive/MyDrive/MLProject/Dataset/Models/"

# LOGISTIC REGRESSION
LR_pickle = "%sLR_model.pkl" % Model_DIR

with open(LR_pickle, 'wb') as file:
    pickle.dump(LR_pipeline, file)

# Linear SVC
# This method uses one vs rest stretegy

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline

LSVC_pipeline = make_pipeline(MinMaxScaler(), LinearSVC(multi_class='ovr', max_iter=2000))
LSVC_pipeline.fit(x_train[:100000], y_train[:100000])

LSVC_y_pred = LSVC_pipeline.predict(x_test[:10000])

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

LSVC_test_r2 = r2_score(y_test[:10000], LSVC_y_pred)
# KNN_train_r2 = r2_score(y_train, KNN_y_train_pred)

LSVC_test_mse = mean_squared_error(y_test[:10000], LSVC_y_pred)
# KNN_train_mse = mean_squared_error(y_train, KNN_y_train_pred)

LSVC_test_accuracy = accuracy_score(y_test[:10000], LSVC_y_pred)
# KNN_train_accuracy = accuracy_score(y_train, KNN_y_train_pred)

print("Test r2_score for LSVC %.2f" % LSVC_test_r2)
# print("Train r2_score %.2f" % KNN_train_r2)

print("\nTest mean squared error for LSVC %.2f" % LSVC_test_mse)
# print("Train mean squared error %.2f" % KNN_train_mse)

print("\nTest accuracy score for LSVC %.2f" % LSVC_test_accuracy)
# print("Train accuracy score %.2f" % KNN_train_accuracy)

print("Classificaion report")
print(classification_report(y_test[:10000], LSVC_y_pred))

conf_mat = confusion_matrix(y_test[:10000], LSVC_y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=dataset['Composer'].unique(), 
            yticklabels=dataset['Composer'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LSVC\n", size=16);

# saving model
import pickle
DIR = "./gdrive/MyDrive/MLProject/Dataset/Models/"

# KNN
LSVC_pickle = "%sLSVC_model.pkl" % Model_DIR

with open(LSVC_pickle, 'wb') as file:
    pickle.dump(LSVC_pipeline, file)

# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

KNN_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())
KNN_pipeline.fit(x_train, y_train)

# Predictions
KNN_y_pred = KNN_pipeline.predict(x_test)

# due to very large dataset this is taking more than 2 hours
#KNN_y_train_pred = KNN_pipeline.predict(x_train)

# calculating other require values
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

KNN_test_r2 = r2_score(y_test, KNN_y_pred)
# KNN_train_r2 = r2_score(y_train, KNN_y_train_pred)

KNN_test_mse = mean_squared_error(y_test, KNN_y_pred)
# KNN_train_mse = mean_squared_error(y_train, KNN_y_train_pred)

KNN_test_accuracy = accuracy_score(y_test, KNN_y_pred)
# KNN_train_accuracy = accuracy_score(y_train, KNN_y_train_pred)

print("Test r2_score for KNN %.2f" % KNN_test_r2)
# print("Train r2_score %.2f" % KNN_train_r2)

print("\nTest mean squared error for KNN %.2f" % KNN_test_mse)
# print("Train mean squared error %.2f" % KNN_train_mse)

print("\nTest accuracy score for KNN %.2f" % KNN_test_accuracy)
# print("Train accuracy score %.2f" % KNN_train_accuracy)

print("Classificaion report")
print(classification_report(y_test, KNN_y_pred))

conf_mat = confusion_matrix(y_test, KNN_y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=dataset['Composer'].unique(), 
            yticklabels=dataset['Composer'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - KNN\n", size=16);

# saving model
import pickle
DIR = "./gdrive/MyDrive/MLProject/Dataset/Models/"

# KNN
KNN_pickle = "%sKNN_model.pkl" % Model_DIR

with open(KNN_pickle, 'wb') as file:
    pickle.dump(KNN_pipeline, file)

# Multinominal naive bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

MNB_pipeline = make_pipeline(MinMaxScaler(), MultinomialNB())
MNB_pipeline.fit(x_train, y_train)

MNB_y_pred = MNB_pipeline.predict(x_test)
MNB_y_train_pred = MNB_pipeline.predict(x_train)

# calculating other require values
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

MNB_test_r2 = r2_score(y_test, MNB_y_pred)
MNB_train_r2 = r2_score(y_train, MNB_y_train_pred)

MNB_test_mse = mean_squared_error(y_test, MNB_y_pred)
MNB_train_mse = mean_squared_error(y_train, MNB_y_train_pred)

MNB_test_accuracy = accuracy_score(y_test, MNB_y_pred)
MNB_train_accuracy = accuracy_score(y_train, MNB_y_train_pred)

print("Test r2_score %.2f" % MNB_test_r2)
print("Train r2_score %.2f" % MNB_train_r2)

print("\nTest mean squared error %.2f" % MNB_test_mse)
print("Train mean squared error %.2f" % MNB_train_mse)

print("\nTest accuracy score %.2f" % MNB_test_accuracy)
print("Train accuracy score %.2f" % MNB_train_accuracy)

print("Classificaion report")
print(classification_report(y_test, MNB_y_pred))

conf_mat = confusion_matrix(y_test, MNB_y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=dataset['Composer'].unique(), 
            yticklabels=dataset['Composer'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - MNB\n", size=16);

# saving model
import pickle
DIR = "./gdrive/MyDrive/MLProject/Dataset/Models/"

# KNN
MNB_pickle = "%sMNB_model.pkl" % Model_DIR

with open(MNB_pickle, 'wb') as file:
    pickle.dump(MNB_pipeline, file)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RDC_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
RDC_pipeline.fit(x_train, y_train)

RDC_y_pred = RDC_pipeline.predict(x_test)
RDC_y_train_pred = RDC_pipeline.predict(x_train)

# calculating other require values
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

RDC_test_r2 = r2_score(y_test, RDC_y_pred)
RDC_train_r2 = r2_score(y_train, RDC_y_train_pred)

RDC_test_mse = mean_squared_error(y_test, RDC_y_pred)
RDC_train_mse = mean_squared_error(y_train, RDC_y_train_pred)

RDC_test_accuracy = accuracy_score(y_test, RDC_y_pred)
RDC_train_accuracy = accuracy_score(y_train, RDC_y_train_pred)

print("Test r2_score %.2f" % RDC_test_r2)
print("Train r2_score %.2f" % RDC_train_r2)

print("\nTest mean squared error %.2f" % RDC_test_mse)
print("Train mean squared error %.2f" % RDC_train_mse)

print("\nTest accuracy score %.2f" % RDC_test_accuracy)
print("Train accuracy score %.2f" % RDC_train_accuracy)

print("Classificaion report")
print(classification_report(y_test, RDC_y_pred))

conf_mat = confusion_matrix(y_test, RDC_y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=dataset['Composer'].unique(), 
            yticklabels=dataset['Composer'].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - RDC\n", size=16);

# saving model
import pickle


RDC_pickle = "%sRDC_model.pkl" % Model_DIR

with open(RDC_pickle, 'wb') as file:
    pickle.dump(RDC_pipeline, file)

# loading all models
DIR = "./drive/MyDrive/MLProject/Dataset/Models/"

import pickle 

LR_pickle = "%sLR_model.pkl" % Model_DIR
MNB_pickle = "%sMNB_model.pkl" % Model_DIR
KNN_pickle = "%sKNN_model.pkl" % Model_DIR
RDC_pickle = "%sRDC_model.pkl" % Model_DIR
LSVC_pickle = "%sLSVC_model.pkl" % Model_DIR

with open(LR_pickle, 'rb') as file:
    LR_pipeline = pickle.load(file)

with open(MNB_pickle, 'rb') as file:
    MNB_pipeline = pickle.load(file)

with open(KNN_pickle, 'rb') as file:
    KNN_pipeline = pickle.load(file)

with open(RDC_pickle, 'rb') as file:
    RDC_pipeline = pickle.load(file)

with open(LSVC_pickle, 'rb') as file:
    LSVC_pipeline = pickle.load(file)

"""## Comparing models with each other
1. Logistic regression
2. KNN
3. Multinomial Naive bayes
4. Random Forest

### Making dataset of f1_score, precision, recall, r2_score, accuracy for each model
"""

model_names = [ "Logistic Regression", "K Nearest Neighbour", "Multinomial Naive Bayes", "Random Forest Classifier", "Linear SVM" ]
models = [LR_pipeline, KNN_pipeline, MNB_pipeline, RDC_pipeline, LSVC_pipeline]

from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, auc, f1_score

r2_scores = []
accuracies = []
precisions = []
recalls = []
aucs = []
f1_scores = []

for idx, (name, model) in enumerate(zip(model_names, models)):
    y_pred = model.predict(x_test)

    r2_scores.append   ( r2_score        (y_test, y_pred) )
    accuracies.append  ( accuracy_score  (y_test, y_pred) )
    f1_scores.append   ( f1_score        (y_test, y_pred, average='weighted') )
    precisions.append  ( precision_score (y_test, y_pred, average='weighted') )
    recalls.append     ( recall_score    (y_test, y_pred, average='weighted') )
    
    print(f"Model: {name}")

score_dict = {
    "MLA": model_names,
    "r2_score": r2_scores,
    "accuracy": accuracies,
    "precision": precisions,
    "recall": recalls,
    'f1_score': f1_scores
}

model_df = pd.DataFrame.from_dict(score_dict)

model_df.to_csv("./drive/MyDrive/MLProject/Dataset/model_df.csv")

model_df = pd.read_csv("./drive/MyDrive/MLProject/Dataset/model_df.csv")
model_df.head()

"""### Accuracy of each model"""

plt.figure(figsize=(12, 6))

sns.barplot(
    x=model_df['MLA'], 
    y=(model_df['accuracy']*100),
    palette="Wistia",
    order=model_df.sort_values('accuracy').MLA
)

for idx, acc in enumerate( model_df.sort_values('accuracy')['accuracy'] ):
    text = '%.2f%%' % (acc * 100)
    plt.text((idx - 0.13), 2, text, fontsize=14)

plt.suptitle("Model Accuracy Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Model Name", labelpad=15, fontsize=14)
plt.ylabel("Accuracy in %", labelpad=15, fontsize=14)
plt.show()

plt.figure(figsize=(12, 6))

sns.barplot(
    x=model_df['MLA'], 
    y=model_df['r2_score'],
    palette="cool",
    order=model_df.sort_values('r2_score').MLA
)

for idx, r2 in enumerate( model_df.sort_values('r2_score')['r2_score'] ):
    text = '%.2f' % r2
    plt.text((idx - 0.15), 0.03, text, fontsize=14)

plt.suptitle("$Model\ R^2\ Score\ Comparison$", fontsize=16, fontweight='bold')
plt.xlabel("Model Name", labelpad=15, fontsize=14)
plt.ylabel("$R^2$ Score", labelpad=15, fontsize=14)
plt.show()

plt.figure(figsize=(12, 6))

sns.barplot(
    x=model_df['MLA'], 
    y=model_df['precision'] * 100,
    palette="gist_earth",
    order=model_df.sort_values('precision').MLA
)

for idx, p in enumerate( model_df.sort_values('precision')['precision'] ):
    text = '%.2f%%' % (p * 100)
    plt.text((idx - 0.13), 2, text, fontsize=14)

plt.suptitle("Model Precision Score Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Model Name", labelpad=15, fontsize=14)
plt.ylabel("Precision Score", labelpad=15, fontsize=14)
plt.show()

plt.figure(figsize=(12, 6))

sns.barplot(
    x=model_df['MLA'], 
    y=model_df['recall'] * 100,
    palette="cool_r",
    order=model_df.sort_values('recall').MLA
)

for idx, p in enumerate( model_df.sort_values('recall')['recall'] ):
    text = '%.2f%%' % (p * 100)
    plt.text((idx - 0.13), 2, text, fontsize=14)

plt.suptitle("Model Recall Score Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Model Name", labelpad=15, fontsize=14)
plt.ylabel("Recall Score", labelpad=15, fontsize=14)
plt.show()

plt.figure(figsize=(12, 6))

sns.barplot(
    x=model_df['MLA'], 
    y=model_df['f1_score'] * 100,
    palette="Wistia",
    order=model_df.sort_values('f1_score').MLA
)

for idx, p in enumerate( model_df.sort_values('f1_score')['f1_score'] ):
    text = '%.2f%%' % (p * 100)
    plt.text((idx - 0.13), 2, text, fontsize=14)

plt.suptitle("Model Recall Score Comparison", fontsize=16, fontweight='bold')
plt.xlabel("Model Name", labelpad=15, fontsize=14)
plt.ylabel("F1 Score", labelpad=15, fontsize=14)
plt.show()

