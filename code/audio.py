"""

 ~ Neurorack project ~
 Audio : Class for the audio handling

 This class contains all audio-processing stuff in the Neurorack.
     - Instantiates the deep model
     - Provides callbacks for playing
         play_noise
         play_model

 Author               :  Ninon Devis, Philippe Esling, Martin Vert
                        <{devis, esling}@ircam.fr>

 All authors contributed equally to the project and are listed aphabetically.

"""
import time

import numpy as np
import sounddevice as sd
import librosa
from parallel import ProcessInput
# from models.ddsp import DDSP
# from models.nsf_impacts import NSF
from models.rave import RAVE
from multiprocessing import Event, Process
from config import config



class Audio(ProcessInput):
    '''
        The Audio class handles every aspect related to audio generation.
        It is based on the ProcessInput system for multiprocessing
    '''

    def __init__(self,
                 callback: callable,
                 model: str,
                 sr: int = 22050):
        '''
            Constructor - Creates a new instance of the Audio class.
            Parameters:
                callback:   [callable]
                            Outside function to call on audio event
                model:      [str], optional
                            Specify the audio model to load
                sr:         int, optional
                            Specify the sampling rate
        '''
        super().__init__('audio')
        # Setup audio callback
        self._callback = callback
        # Create our own event signal
        self._signal = Event()
        # Configure audio
        self._sr = sr
        # Set devices default
        self.set_defaults()
        self._model_name = model
        # Current block stream
        self._cur_stream = None
        # Set model
        self.load_model()
        # frame length
        self.frame_len = 8192
        # For the sinewave & sample play
        self.start_idx = 0
        self.tmp_flag = 0
        
        self.sample, sr = librosa.load("data/Alborosie.wav", sr=self._sr, duration=10.0)
        print("loaded sample, normalized from", np.amax(self.sample))
        self.sample = self.sample / np.amax(self.sample)
        print(self.sample.shape, sr)
        self.max_idx = (self.sample.shape[0] // self.frame_len) * self.frame_len
        print(self.max_idx, "max idx")

    def load_model(self):
        # if self._model_name == 'ddsp':
        #     self._model = DDSP()
        # elif self._model_name == 'nsf':
        #     self._model = NSF()
        if self._model_name == 'rave':
            self._model = RAVE()
        else:
            raise NotImplementedError

    def callback(self, state, queue):
        # First perform a model burn-in
        # print('Performing model burn-in')
        state["audio"]["mode"].value = config.audio.mode_burnin
        self._model.preload()
        # Then switch to wait (idle) mode
        print('Audio ready')
        state["audio"]["mode"].value = config.audio.mode_idle
        # Perform display loop
        while True:
            self._signal.wait()
            if self._signal.is_set():
                # The refresh comes from an external signal
                self._signal.clear()
                self.handle_signal_event(state)

    def handle_signal_event(self, state):
        cur_event = state["audio"]["event"]
        if cur_event in [config.events.gate0]:
            self.play_model_block(state)

    def set_defaults(self):
        '''
            Sets default parameters for the soundevice library.
            See
        '''
        sd.default.samplerate = self._sr
        sd.default.device = 1
        sd.default.latency = 'low'
        sd.default.dtype = 'float32'
        sd.default.blocksize = 0
        sd.default.clip_off = False
        sd.default.dither_off = False
        sd.default.never_drop_input = False


    def get_sin(self, frames, amplitude=1.0, frequency=440.0):
        t = (self.start_idx + np.arange(frames)) / self._sr
        t = t.reshape(-1, 1)
        self.start_idx += frames
        tmp = amplitude * np.sin(2 * np.pi * frequency * t)
        tmp = tmp.swapaxes(0, 1)
        tmp = np.expand_dims(tmp, 0)
        return tmp

    def play_model_block(self, state, wait: bool = True):
        def callback_block(outdata, frames, time_c, status):
            start_time = time.time()
            # # sin = self.get_sin(frames) 

            ####################
            # Sample generation
            sample = self.sample[self.start_idx:self.start_idx + frames]  # 154336,
            self.start_idx += frames
            if self.start_idx >= self.max_idx:
                self.start_idx = 0
                print("sample looped")
            sample = np.expand_dims(sample, 0)
            sample = np.expand_dims(sample, 0)
            lats = self._model.encode(sample)
            print(lats.shape)
            # if state['cv_active'][2]:
                # print("*", state['cv'][2], "for lat 0")
                # lats[0][0] *= state['cv'][2] 
            if state['cv_active'][3]:
                print("*", state['cv'][3], "for lat 1")
                lats[0][1] *= state['cv'][3] 
            if state['cv_active'][4]:
                lats[0][2] *= state['cv'][4] 
            if state['cv_active'][5]:
                lats[0][3] *= state['cv'][5] 
            cur_data = self._model.decode(lats)[0]

            # # print("input shape", sample.shape)  # Must be [1, 1, frames]
            # # cur_data = self.sample[self.start_idx:self.start_idx + frames]
            # print("output shape", cur_data.shape)  # Must be [frames]
            
            # cur_data = self._model.generate_random(2)
            
            # outdata[:] = np.random.rand(frames, 1)
            # print("noise")

            if cur_data is None:
                raise sd.CallbackStop()
            outdata[:] = cur_data[:, np.newaxis]
            print("time:", time.time() - start_time)

        if self._cur_stream == None:
            self._cur_stream = sd.OutputStream(blocksize=self.frame_len, callback=callback_block, channels=1, samplerate=self._sr)
            self._cur_stream.start()
            print('Stream launched')
        elif not self._cur_stream.active:
            print('Restart stream')
            self._cur_stream.close()
            self._cur_stream = sd.OutputStream(blocksize=self.frame_len, callback=callback_block, channels=1, samplerate=self._sr)
            self._cur_stream.start()
    
    ###########################################
    
    def play_model(self, state, wait: bool = True):
        '''
            Play some random noise of a given length for checkup.
            Parameters:
                wait:       [bool], optional
                            Wait on the end of the playback
        '''
        print("play_model start")
        state["audio"]["mode"].value = config.audio.mode_play
        audio = self._model.generate_prior_random(24*5)
        
        # print(audio)
        # print(f"min: {np.min(audio)} / max: {np.max(audio)}")
        print('generation ended')
        print(audio.shape)
        sd.play(audio, self._sr)
        if wait:
            self.wait_playback()
        state["audio"]["mode"].value = config.audio.mode_idle
        print("play_model end")

    def play_noise(self, wait: bool = True, length: int = 2):
        '''
            Play some random noise of a given length for checkup.
            Parameters:
                wait:       [bool], optional
                            Wait on the end of the playback
                length:     [float], optional
                            Length of signal to generate (in seconds)
        '''
        print("start noise")
        audio = np.random.randn(length * self._sr)
        print(f"min: {np.min(audio)} / max: {np.max(audio)}")
        print(audio.shape)
        sd.play(audio, self._sr)
        if (wait):
            self.wait_playback()
        print("end noise")

    def play_sine_block(self, amplitude=1.0, frequency=440.0):
        '''
            Play a sinus signal
            Parameters:
                amplitude:  [float], optional
                            Amplitude of the sinusoid
                length:     [int], optional
                            Length of signal to generate (in seconds)
        '''
        print("start sine")
        def callback(outdata, frames, time, status):
            if status:
                print(status)
                print('')
            global start_idx
            t = (start_idx + np.arange(frames)) / self._sr
            t = t.reshape(-1, 1)
            outdata[:] = amplitude * np.sin(2 * np.pi * frequency * t)
            start_idx += frames

        with sd.OutputStream(device=sd.default.device, channels=1, callback=callback,
                             samplerate=self._sr):
            input()
        print("end sine")

    def stop_playback(self):
        ''' Stop any ongoing playback '''
        sd.stop()

    def wait_playback(self):
        ''' Wait on eventual playback '''
        sd.wait()

    def get_status(self):
        ''' Get info about over/underflows (play() or rec()) '''
        return sd.get_status()

    def get_stream(self):
        ''' Get a reference to the current stream (play() or rec()) '''
        return sd.get_stream()

    def query_devices(self):
        ''' Return information about available devices '''
        return sd.query_devices()

    def query_hostapis(self):
        ''' Return information about host APIs '''
        return sd.query_hostapis()


if __name__ == '__main__':
    audio = Audio(None)
    audio.model_burn_in()
    audio._signal.wait(4)
    print('Starting play')
    audio.play_model_block(None)
    audio._signal.wait(1000)
