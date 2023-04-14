import librosa
import numpy as np
from PIL import Image, ImageOps
import os
import soundfile as sf

image_folder = 'results/test1'
wav_folder = 'results/test1'

os.makedirs(wav_folder, exist_ok=True)

window_length = 511
hop_length = 128

'''for filename in os.listdir(image_folder):
    image = Image.open(os.path.join(image_folder, filename))
    image_array = np.asarray(image)
    image_array = np.transpose(image_array, (2,0,1))
    

    amplitude = librosa.db_to_amplitude(image_array[0], ref=0.00001)
    phase = ((image_array[1]/255)*2*np.pi)-np.pi

    spectrogram = amplitude*np.exp(1j*phase)
    audio = librosa.istft(spectrogram, hop_length=hop_length, n_fft=window_length)

    sf.write(os.path.join(wav_folder, filename.replace('.png', '.wav')), audio, 11025)
    break'''

# amplitude only

for filename in os.listdir(image_folder):
    image = Image.open(os.path.join(image_folder, filename))
    image_array = np.asarray(image)
    image_array = image_array[:,:,0]
    #image_array = np.transpose(image_array, (2,0,1))
    

    amplitude = librosa.db_to_amplitude(image_array, ref=0.00001)
    #phase = ((image_array[1]/255)*2*np.pi)-np.pi

    spectrogram = amplitude*np.exp(1j*phase)
    audio = librosa.istft(amplitude, hop_length=hop_length, n_fft=window_length)

    sf.write(os.path.join(wav_folder, filename.replace('.png', '.wav')), audio, 11025)
    break