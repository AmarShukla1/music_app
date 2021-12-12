import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import librosa
import math
from tensorflow import keras
from tensorflow.keras.models import load_model
from numba.core.types.functions import MakeFunctionLiteral
from home_app.models import contact
from django.shortcuts import render
from datetime import datetime
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

import numpy as np
import os
from django.conf import settings

# Views


def index(request):
    context = {
        'variable': "amar"

    }
    return render(request, 'index.html', context)


def about(request):
    return render(request, 'about.html')


def contacts(request):
    if request.method == "POST":

        name = request.POST.get('name')

        email = request.POST.get('email')

        phone = request.POST.get('phone')

        desc = request.POST.get('desc')
        Contact = contact(name=name, email=email, phone=phone,
                          desc=desc, date=datetime.today())
        Contact.save()
    return render(request, 'contact.html')


def results(request):
    data = []
    SAMPLE_RATE = 22050
    n_mfcc = 13
    n_fft = 2048
    hop_length = 512
    num_segments = 10
    media_path = settings.MEDIA_ROOT
    file_path = os.path.join(media_path, "../media")

    files = os.listdir(file_path)
    for i in range(0, len(files)):
        files[i] = './media/'+files[i]
    files = sorted(files, key=os.path.getmtime)
    print(files)
    main_file = files[len(files)-1]
    print(main_file)
    signal, sr = librosa.load(main_file, sr=SAMPLE_RATE)
    duration = librosa.get_duration(signal, sr=SAMPLE_RATE)

    # full SAMPLE RATE of the track
    SAMPLES_PER_TRACK = SAMPLE_RATE * duration
    # number of segments in the track
    num_samples_per_segment = SAMPLES_PER_TRACK // num_segments

    # expected mfcc size for a segment
    expected_num_mfcc_vector_per_segment = math.ceil(
        num_samples_per_segment / hop_length)

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

        if(len(mfcc) == expected_num_mfcc_vector_per_segment):
            data.append(mfcc.tolist())

    model = pickle.load(open("KNN_model.pkl", "rb"))
    prediction = model.predict(data[0])

    index = np.bincount(prediction).argmax()
    print('amar')
    name = ['schubert', 'mozart', 'haydn',
            'bach', 'brahms', 'dvorak', 'beethoven']
    image = name[index]

    # return render(request,'results.html',{'results':prediction})

    return render(request, 'classical.html', {'results': name[index], 'image': f'static/{image}.jpg'})
    # there is some mistake in predict function will make it correct..


# 6. Dvorak
# 7. Beethoven
#  5. Brahms
#   4. Bach
#   3. Haydn -
#   2. Mozart
#   1.   Schubert
def genres(request):
    media_path = settings.MEDIA_ROOT
    file_path = os.path.join(media_path, "../media")

    files = os.listdir(file_path)
    for i in range(0, len(files)):
        files[i] = './media/'+files[i]
    files = sorted(files, key=os.path.getmtime)

    main_file = files[len(files)-1]
    print(main_file)
    y, sr = librosa.load(main_file, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    song = [np.mean(chroma_stft), np.mean(rmse), np.mean(
        spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
    for e in mfcc:
        song.append(np.mean(e))
    
    a = np.array([song])
    scaler = pickle.load(open("scaler.pkl", "rb"))
    a = scaler.transform(a)
    print(a)
    loaded_model = load_model('network.h5')
    prediction = loaded_model.predict(a)
    print(prediction)
    name = ['blues', 'classical', 'country', 'disco',
            'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    index = name[np.argmax(prediction[0])]
    
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="d62b0a8580f14891b00ed399599e9a29",
                                                               client_secret="b9f08100daaa4df9a1ab020b6583fd94"))
    query=f"genre:{index} year:2019"
    print(query)
    
    track = sp.search(query, type="track")


    

    tracks = track['tracks']
    tempst = set()
    finalans = []
    for item in tracks['items']:
        temp = [item['name'], item['album']['artists'][0]['name'], item['popularity'],
            item['preview_url'], item['external_urls']['spotify'], item['album']['images'][1]['url']]
        if(item['id'] not in tempst):
            tempst.add(item['id'])
            finalans.append(temp)
    return render(request, 'genres.html', {'results': index, 'image': f'static/{index}.jpg','links':finalans})


def spotify(request):
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="d62b0a8580f14891b00ed399599e9a29",
                                                               client_secret="b9f08100daaa4df9a1ab020b6583fd94"))
    query=request.POST.get('search','')
    
    
    track = sp.search(query, type="track")


    

    tracks = track['tracks']
    tempst = set()
    finalans = []
    for item in tracks['items']:
        temp = [item['name'], item['album']['artists'][0]['name'], item['popularity'],
            item['preview_url'], item['external_urls']['spotify'], item['album']['images'][1]['url']]
        if(item['id'] not in tempst):
            tempst.add(item['id'])
            finalans.append(temp)
    
    return render(request, 'spotify.html',{'links':finalans})