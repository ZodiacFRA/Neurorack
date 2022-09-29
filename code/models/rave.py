import time
import sounddevice as sd
""" 
I import torch in preload() instead of here otherwise torch is
imported by the parent process as well, and cuda is not setup for multiprocessing
so I import it inside a class variable
"""

class RAVE():
    def __init__(self):
        self.model = None
        self.torch = None
        self.m_path = "./models/nasa.ts"
        self.f_pass = 3

    def preload(self):
        print('Loading torch')
        import torch
        self.torch = torch
        self.torch.backends.cudnn.benchmark = True

        print("Loading RAVE model")
        self.model = self.torch.jit.load(self.m_path, map_location="cuda")
        print("RAVE model loaded")
        self.burn_in()

    def generate_prior_random(self, length=48):
        # length 1 = 2048, 24 ~= 1sec
        # print('Prior generation, random, length ' + str(length))
        # Batch, Channels, length
        x = self.torch.randn(1, 1, length).cuda()
        with self.torch.no_grad():
            # 1 temperature value for each time step, outputs [n_latents, length]
            lat = self.model.prior(x)
            # Decode
            audio = self.model.decode(lat)
        # Multiply by 10 to get the proper volume
        tmp = audio.squeeze(0).squeeze(-1).cpu()[0]
        amp = self.torch.max(tmp) - self.torch.min(tmp)
        if amp == 0:
            amp = 0.01
        tmp = self.torch.mul(tmp, 10 / amp)
        print(tmp.shape)
        print(f"Model generation min: {self.torch.min(tmp)} / max: {self.torch.max(tmp)}")
        return tmp

    def generate_random(self, length=48):
        # length 1 = 2048, 24 ~= 1sec
        # print('generation, random, length ' + str(length))
        # Batch, Channels, length
        lat = self.torch.randn(1, 8, length).cuda()
        with self.torch.no_grad():
            # 1 temperature value for each time step, outputs [n_latents, length]
            # Decode
            audio = self.model.decode(lat)
        # Multiply by 10 to get the proper volume
        tmp = audio.squeeze(0).squeeze(-1)[0]
        amp = self.torch.max(tmp) - self.torch.min(tmp)
        if amp == 0:
            amp = 0.01
        tmp = self.torch.mul(tmp, 2)
        print(tmp.shape)
        print(f"Model generation min: {self.torch.min(tmp)} / max: {self.torch.max(tmp)}")
        return tmp.cpu()

    def forward(self, audio):
        print('forward')
        with self.torch.no_grad():
            # 1 temperature value for each time step, outputs [n_latents, length]
            audio = self.model(audio)
        return audio.squeeze(0).squeeze(-1)

    def burn_in(self):
        for p in range(self.f_pass):
            print("Starting RAVE preload pass", p, self.f_pass)
            cur_time = time.monotonic()
            with self.torch.no_grad():
                x = self.torch.randn(1, 1, 1).cuda()
                audio = self.model(x)
            print('Time : ' + str(time.monotonic() - cur_time))


if __name__ == '__main__':
    rave = RAVE()
    rave.preload()
    print(rave.generate_prior_random())
