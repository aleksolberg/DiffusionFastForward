import librosa
import os
import numpy as np
from PIL import Image, ImageOps
import soundfile as sf

train_folder = 'datasets/randomMIDI/PianoViolin11025/WAV/foreground/train/mix'
val_folder = 'datasets/randomMIDI/PianoViolin11025/WAV/foreground/val/mix'
test_folder = 'datasets/randomMIDI/PianoViolin11025/WAV/foreground/test/mix'

in_folders = [train_folder, val_folder, test_folder]

train_folder_out = 'datasets/randomMIDI/PianoViolin11025/jpeg_amp_only/train/mix'
val_folder_out = 'datasets/randomMIDI/PianoViolin11025/jpeg_amp_only/val/mix'
test_folder_out = 'datasets/randomMIDI/PianoViolin11025/jpeg_amp_only/test/mix'

out_folders = [train_folder_out, val_folder_out, test_folder_out]

os.makedirs(train_folder_out, exist_ok=True)
os.makedirs(val_folder_out, exist_ok=True)
os.makedirs(test_folder_out, exist_ok=True)

# STFT parameters:
window_length = 511
hop_length = 128

for i in range(len(in_folders)):
    for filename in os.listdir(in_folders[i]):
        audio = librosa.load(os.path.join(in_folders[i], filename), sr=11025)
        spectrogram = librosa.stft(audio[0], n_fft=window_length, hop_length=hop_length)

        # Preserving phase info in second channel
        amplitude = np.abs(spectrogram)
        phase = np.angle(spectrogram)

        spectrogram = np.stack((amplitude, phase))

        # Cutting the spectrogram to be square, converting to DB, normalizing
        spectrogram = spectrogram[:, :, :256]
        spectrogram[0] = librosa.amplitude_to_db(spectrogram[0], ref=0.00001)
        spectrogram[1] = (spectrogram[1]+np.pi)/(2*np.pi)*255
        spectrogram = np.transpose(spectrogram, (1,2,0)).astype(np.uint8)

        image = Image.fromarray(spectrogram[:,:,0])
        image.save(os.path.join(out_folders[i], filename.replace('.wav', '.png')))
