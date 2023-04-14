import librosa
import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SpectrogramDataset(Dataset):
    """Dataset returning spectrograms from audio in a folder."""

    def __init__(self,
                 target_dir,
                 condition_dir=None,
                 transforms=None,
                 paired=True,
                 return_pair=False,
                 sample_rate = 11025,
                 window_length=511,
                 hop_length=128,
                 out_shape = (3, 256, 256)):
        self.target_dir = target_dir
        self.condition_dir = condition_dir
        self.transforms = transforms
        self.return_pair=return_pair

        self.sample_rate = sample_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.out_shape=out_shape
        
        # check files
        supported_formats=['wav', 'mp3']        
        self.files=[el for el in os.listdir(self.target_dir) if el.split('.')[-1] in supported_formats]
        if condition_dir is not None:
            self.condition_files=[el for el in os.listdir(self.condition_dir) if el.split('.')[-1] in supported_formats]
            assert self.files == self.condition_files, 'Filenames do not match'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_name = os.path.join(self.target_dir, self.files[idx])
        target_audio, sr = librosa.load(target_name, sr=self.sample_rate)
        target_spectrogram = librosa.stft(target_audio, n_fft=self.window_length, hop_length=self.hop_length)

        target_amplitude = torch.from_numpy(librosa.amplitude_to_db(np.abs(target_spectrogram), ref=1e-5)/255)
        target_phase = torch.from_numpy(np.angle(target_spectrogram))

        # Shaping tensors
        target_amplitude = target_amplitude[:self.out_shape[1], :self.out_shape[2]].unsqueeze(0).repeat(self.out_shape[0], 1, 1)
        target_phase = target_phase[:self.out_shape[1], :self.out_shape[2]].unsqueeze(0).repeat(self.out_shape[0], 1, 1)

        if self.condition_dir is not None:
            condition_name = os.path.join(self.condition_dir, self.files[idx])
            condition_audio, sr = librosa.load(condition_name, sr=self.sample_rate)
            condition_spectrogram = librosa.stft(condition_audio, n_fft=self.window_length, hop_length=self.hop_length)

            condition_amplitude = torch.from_numpy(librosa.amplitude_to_db(np.abs(condition_spectrogram), ref=1e-5)/255)

            condition_phase = torch.from_numpy(np.angle(condition_spectrogram))

            # Shaping tensors
            condition_amplitude = condition_amplitude[:self.out_shape[1], :self.out_shape[2]].unsqueeze(0).repeat(self.out_shape[0], 1, 1)
            condition_phase = condition_phase[:self.out_shape[1], :self.out_shape[2]].unsqueeze(0).repeat(self.out_shape[0], 1, 1)

        target = {'name':self.files[idx], 'amplitude': target_amplitude, 'phase': target_phase}
        condition = {'name':self.condition_files[idx], 'amplitude': condition_amplitude, 'phase': condition_phase}

        if self.return_pair:
            return condition, target
        else:
            return target
        
    def to_audio(self, spectrogram_dict):
        ' Takes a dictionary of the form {name, amplitude, phase} and converts to audio'

        amplitude = torch.mean(spectrogram_dict['amplitude'], dim=0) * 255
        amplitude = librosa.db_to_amplitude(amplitude, ref=1e-5).numpy()
        phase = torch.mean(spectrogram_dict['phase'], dim=0).numpy()

        spectrogram = amplitude*np.exp(1j*phase)
        audio = librosa.istft(spectrogram, hop_length=self.hop_length, n_fft=self.window_length)

        return audio


    def save_audio(self, spectrogram_dict, output_folder = 'results'):
        audio = self.to_audio(spectrogram_dict)

        sf.write(os.path.join(output_folder, spectrogram_dict['name']), audio, self.sample_rate)
        return audio