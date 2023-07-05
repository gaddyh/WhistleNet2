import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import wave
import scipy
import math
from IPython.display import Audio
import random

fr=48000
top_db = 20
mels=256

def length_sr(y1):
  if( y1.shape[0] < fr):
    zeros = math.floor((fr - y1.shape[0])/2)
    y1 = np.pad(y1, (zeros, zeros), 'constant')
  else:
    y1 = y1[:fr]
  return y1

def roll_rand(x):
  prob =  np.random.sample(1)
  shifts = math.floor(prob * fr)
  return np.roll(x, shifts)

def noise_rand(x):
  prob =  np.random.sample(1)
  fhat = np.fft.fft(x, fr)
  fhat *= prob
  yi = np.fft.ifft(fhat).astype(float)
  return yi

''' 
    trim top_db=8
    sr=48000
    n_mels=128
    pass only 2-5khz (fft -> pass -> ifft)
    returns power spec binirized -20db shape:(35,94)
'''
def create_chroma(y):
  a = create_power_spec(y)
  threshold = np.percentile(a, 98.9) 
  print('percentile 98.9 of power spec D2 (threshold): ', threshold)
  D2 = np.select([a <=threshold, a>threshold], [np.zeros_like(a), np.ones_like(a)])
  D2 = D2[40:120,:]
  print('D2 shape: ', D2.shape)
  return D2

def create_power_spec(y):
  fhat = np.fft.fft(y, fr)
  fhat[0:700]=0
  fhat[4000:]=0
  a = fhat.real[:24000]
  print('fhat max:' , a.max())
  print('fhat max index:' , a.argmax())
  fhat[0:a.argmax() - 300] = 0  # when main freq is higher we can remove a higher range of lower frequncies

  yi = np.fft.ifft(fhat).astype(float)

  S2 = librosa.feature.melspectrogram(y=yi, sr=fr, n_mels=mels)
  D2 = librosa.power_to_db(S2, ref=np.max)
  return D2

def make_sample(file):
  y1o, sr = librosa.load(file, sr=fr)
  y1 = librosa.effects.trim(y1o, top_db=top_db)[0]
  y1 = length_sr(y1)
  
  fig, ax = plt.subplots(nrows=2, sharex=True)
  S2 = librosa.feature.melspectrogram(y=y1, sr=fr, n_mels=128)
  D2 = librosa.power_to_db(S2, ref=np.max)
  img = librosa.display.specshow(D2, x_axis='time', y_axis='mel', sr=sr, ax=ax[0])
  fig.colorbar(img, ax=[ax[0]])
  ax[0].label_outer()

  c = create_chroma(y1)
 
  img = librosa.display.specshow(c, x_axis='time', y_axis='mel', sr=sr, ax=ax[1])
  fig.colorbar(img, ax=[ax[1]])

  return Audio(data=y1, rate=sr)

def make_sample_from_audio(y1):
  y1 = librosa.effects.trim(y1, top_db=top_db)[0]
  y1 = length_sr(y1)
  
  fig, ax = plt.subplots(nrows=2, sharex=True)
  S2 = librosa.feature.melspectrogram(y=y1, sr=fr, n_mels=128)
  D2 = librosa.power_to_db(S2, ref=np.max)
  img = librosa.display.specshow(D2, x_axis='time', y_axis='mel', sr=sr, ax=ax[0])
  fig.colorbar(img, ax=[ax[0]])
  ax[0].label_outer()

  c = create_chroma(y1)
 
  img = librosa.display.specshow(c, x_axis='time', y_axis='mel', sr=sr, ax=ax[1])
  fig.colorbar(img, ax=[ax[1]])

  return Audio(data=y1, rate=sr)

def make_sample_for_network(y1):
  y1 = librosa.effects.trim(y1, top_db=top_db)[0]
  y1 = length_sr(y1)
  c = create_chroma(y1)
  return c