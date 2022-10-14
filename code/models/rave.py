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
        self.m_path = "./models/vintage.ts"
        self.f_pass = 3

    def preload(self):
        print('Loading torch')
        import torch
        self.torch = torch
        self.torch.backends.cudnn.benchmark = True

        print("Loading RAVE model")
        self.model = self.torch.jit.load(self.m_path, map_location="cuda")
        print("RAVE model loaded")
        # self.burn_in()

    # def test(self, length=48):
    #     # length 1 = 2048, 24 ~= 1sec
    #     flag = False
    #     if length != 4:
    #         flag = True
    #         length = 4
    #     with self.torch.no_grad():
    #         # 1 temperature value for each time step, outputs [n_latents, length]
    #         if flag:
    #             start_time = time.time()
    #             test = self.model.prior(self.torch.randn(1, 1, length).cuda())
    #             print("time:", time.time() - start_time)
    #         lat = self.torch.randn(1, 8, length).cuda()
    #         print("random", lat.shape)
    #         audio = self.model.decode(lat)
    #         audio = audio.squeeze(0).squeeze(-1)[0]
    #     # print(f"Model generation min: {self.torch.min(audio)} / max: {self.torch.max(audio)}")
    #     return audio.cpu()

    def generate_random(self, length=48):
        # length 1 = 2048, 24 ~= 1sec
        with self.torch.no_grad():
            # 1 temperature value for each time step, outputs [n_latents, length]
            lat = self.torch.randn(1, 8, length).cuda()
            audio = self.model.decode(lat)
            audio = audio.squeeze(0).squeeze(-1)[0]
        # print(f"Model generation min: {self.torch.min(audio)} / max: {self.torch.max(audio)}")
        return audio.cpu()

    def generate_prior(self, length=48):
        # length 1 = 2048, 24 ~= 1sec
        with self.torch.no_grad():
            # 1 temperature value for each time step, outputs [n_latents, length]
            lat = self.model.prior(self.torch.randn(1, 1, length).cuda())
            audio = self.model.decode(lat)
            audio = audio.squeeze(0).squeeze(-1)[0]
        # print(f"Model generation min: {self.torch.min(audio)} / max: {self.torch.max(audio)}")
        return audio.cpu()

    def forward(self, audio):
        with self.torch.no_grad():
            audio = self.model(self.torch.tensor(audio).cuda().float())
            audio = audio.squeeze(0).squeeze(0)
        return audio.cpu()

    def encode(self, audio):
        with self.torch.no_grad():
            lats = self.model.encode(self.torch.tensor(audio).cuda().float())
        return lats

    def decode(self, lats):
        with self.torch.no_grad():
            audio = self.model.decode(lats)
        return audio.cpu().squeeze(0)

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
