import streamlit as st
import librosa
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 22050
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_mfcc(file_path, max_pad_len=1300):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc.T[..., np.newaxis]

model = tf.keras.models.load_model("music_genre_cnn_model.h5")

st.title("🎧 Music Genre Classifier")

file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if file is not None:
    st.audio(file, format='audio/wav')
    with open("temp.wav", "wb") as f:
        f.write(file.read())
    mfcc = extract_mfcc("temp.wav")
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc)
    genre = genres[np.argmax(prediction)]
    confidence = np.max(prediction)
    st.success(f"🎵 Predicted Genre: **{genre}** ({confidence:.2f} confidence)")
