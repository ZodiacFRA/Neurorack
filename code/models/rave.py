import time
import torch
import sounddevice as sd


class RAVE():
    m_path = "./models/merkel_run.ts"
    f_pass = 3

    def __init__(self):
        print('Creating RAVE')
        self.model = torch.jit.load(self.m_path, map_location="cuda")


    def preload(self):
        x = torch.randn(1, 1, 1)
        for p in range(self.f_pass):
            with torch.no_grad():
                audio = self.model(x)

    def play(self, audio):
        tmp = audio.cpu()
        tmp = tmp.squeeze(0)[0].cpu()

        print("play")
        sd.play(tmp, 48000)
        time.sleep(10)
        sd.stop()
        print("stop")

    def generate_prior_random(self, length=200):
        print('Prior generation, random, length ' + str(length))
        # Batch, Channels, length
        x = torch.randn(1, 1, length)
        with torch.no_grad():
            # 1 temperature value for each time step, outputs [n_latents, length]
            lat = self.model.prior(x)
            # Decode
            audio = self.model.decode(lat)

        self.play(audio)

        return audio.squeeze(0).squeeze(-1)

    def generate_random(self, length=200):
        print('Random generation, length ' + str(length))
        lat = torch.randn(1, 1, length)
        with torch.no_grad():
            # 1 temperature value for each time step, outputs [n_latents, length]
            audio = self.model.decode(lat)
        return audio.squeeze(0).squeeze(-1)

    def forward(self, audio):
        print('forward')
        with torch.no_grad():
            # 1 temperature value for each time step, outputs [n_latents, length]
            audio = self.model(audio)

        self.play(audio)

        return audio.squeeze(0).squeeze(-1)


if __name__ == '__main__':
    rave = RAVE()
    rave.preload()
    print(rave.generate_prior_random())
