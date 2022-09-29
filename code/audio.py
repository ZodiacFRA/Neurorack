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
import numpy as np
import sounddevice as sd
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
                 sr: int = 48000):
        '''
            Constructor - Creates a new instance of the Audio class.
            Parameters:
                callback:   [callable]
                            Outside function to call on audio event
                model:      [str], optional
                            Specify the audio model to load
                sr:         int, optional
                            Specify the sampling rate [default: 22050]
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
        print('Performing model burn-in')
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
            # self._model.generate_prior_random(state)
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

    def play_model_block(self, state, wait: bool = True):
        def callback_block(outdata, frames, time, status):
            print('-----------')
            cur_data = self._model.generate_random(1)
            if cur_data is None:
                raise sd.CallbackStop()
            
            # print(cur_data.shape)
            # print(outdata.shape)
            outdata[:] = cur_data[:, np.newaxis]
            # print(outdata.shape)
            # outdata[:] = np.random.rand(2048, 1)

        if self._cur_stream == None:
            self._cur_stream = sd.OutputStream(blocksize=2048, callback=callback_block, channels=1, samplerate=self._sr)
            self._cur_stream.start()
            print('Stream launched')
        elif not self._cur_stream.active:
            print('Restart stream')
            self._cur_stream.close()
            self._cur_stream = sd.OutputStream(blocksize=2048, callback=callback_block, channels=1, samplerate=self._sr)
            self._cur_stream.start()

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
