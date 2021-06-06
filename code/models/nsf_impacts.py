import sklearn
import torch
import librosa
import numpy as np
import time
import os
import tqdm
import torchaudio
import soundfile as sf
import threading

def spectral_features(y, sr):
    features = [None] * 7
    features[0] = librosa.feature.rms(y)
    features[1] = librosa.feature.zero_crossing_rate(y)
    # Spectral features
    S, phase = librosa.magphase(librosa.stft(y=y))
    # Compute all descriptors
    features[2] = librosa.feature.spectral_rolloff(S=S)
    features[3] = librosa.feature.spectral_flatness(S=S)
    features[4] = librosa.feature.spectral_bandwidth(S=S)
    features[5] = librosa.feature.spectral_centroid(S=S)
    features[6] = librosa.yin(y, 50, 5000, sr=sr)[np.newaxis, :]
    features = np.concatenate(features).transpose()
    features[np.isnan(features)] = 1
    features = features[:-1, :]
    return features


class NSF:
    # m_path = "/home/martin/Desktop/Impact-Synth-Hardware/code/models/model_nsf_sinc_ema_impacts_waveform_5.0.th"
    m_path = "/home/hime/Work/Neurorack/Impact-Synth-Hardware/code/models/model_nsf_sinc_ema_impacts_waveform_5.0.th"
    f_pass = 3

    def __init__(self):
        # Testing NSF
        print('Creating empty NSF')
        self._model = None
        self._wav_file = 'reference_impact.wav'
        self._n_blocks = 8
        self._thread = None
        self._last_gen_block = -1
        self.generated_queue = []
        self.generate_end = False

    def dummy_features(self, wav):
        y, sr = librosa.load(wav)
        features = spectral_features(y, sr)
        return features

    def preload(self):
        self._model = torch.load(self.m_path, map_location="cuda")
        self._model = self._model.cuda()
        print("loaded")
        self.features = self.dummy_features(self._wav_file)
        self.features = torch.tensor(features).unsqueeze(0).cuda().float()
        tmp_features = self.features[:, :self._n_blocks, :]
        for p in range(self.f_pass):
            print("pass")
            with torch.no_grad():
                cur_blocks = self._model(tmp_features)

    def generate_random(self, length=200):
        print('Generating random length ' + str(length))
        features = [torch.randn(1, length, 1).cuda()] * 7
        with torch.no_grad():
            audio = self._model(features)
        return audio.squeeze().detach().cpu().numpy()

    def generate(self, features):
        # features = self.dummy_features(wav)
        # features = torch.tensor(features).unsqueeze(0).cuda().float()
        with torch.no_grad():
            audio = self._model(features)
        return audio.squeeze().detach().cpu().numpy()
    
    def generate_thread(self, args):
        if (self.last_gen_block+self._n_blocks + 1) > self.features.shape[1]:
            self.generate_end = True
            return
        cur_feats = self.features[:, self.last_gen_block:(self.last_gen_block+self._n_blocks + 1), :]
        cur_audio = model.generate(cur_feats)
        if (self.last_val is not None):
            cur_audio[:512] = (self.last_val * np.linspace(1, 0, 512)) + (cur_audio[:512] * np.linspace(0, 1, 512))
        self.last_val = cur_audio[-512:]
        cur_audio = cur_audio[:-512]
        for b in range(self._n_blocks):
            self.generate_queue.append(cur_audio[(b * 512):((b+1)*512)])
        self.last_gen_block += self._n_blocks
        
    
    def block_generate(self, block_idx):
        if (block_idx == 0):
            self.generate_queue = []
            self.generate_end = False
        if (len(self.generated_queue) <= block_idx):
            self.generate_thread()
        if (not self._thread or (not self._thread.is_alive()) and (not self.generate_end)):
            self._thread = threading.Thread(target=self.generate_thread, args=(1,))
            self._thread.start()
        return self.generated_queue[block_idx]

if __name__ == '__main__':
    root_dir = "/home/hime/Work/dataset/toydataset"
    wav_adresses = [files_names for files_names in os.listdir(root_dir) if
                    (files_names.endswith('.wav') or files_names.endswith('.mp3'))]
    model = NSF()
    model.preload()
    for wav in wav_adresses:
        y, sr = librosa.load(root_dir + '/' + wav)
        features = spectral_features(y, sr)
        print(features.shape)
        features = torch.tensor(features).unsqueeze(0).cuda().float()
        audio = model.generate(features)
        sf.write("generate" + str(wav) + ".wav", audio, sr)


