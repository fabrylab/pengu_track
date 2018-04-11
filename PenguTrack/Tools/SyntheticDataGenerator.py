import numpy as np
from PenguTrack.Detectors import Measurement

class SyntheticDataGenerator(object):
    def __init__(self, object_number, position_variation, speed_variation, model, loose=False):
        self.N = int(object_number)
        self.R = float(position_variation)
        self.Q = float(speed_variation)
        self.Model = model
        self.Loose = bool(loose)
        vec = np.random.rand(self.N, self.Model.State_dim, 1)
        vec[:, ::2] *= self.R
        vec[1:, ::2] *= self.Q
        self.Objects = {-1: dict(zip(range(self.N), vec))}

        # for i in range(100):
        #     self.step()

    def step(self):
        i = max(self.Objects.keys())
        tracks = list(self.Objects[i].keys())
        positions = [self.Objects[i][k] for k in tracks]
        self.Objects[i+1] = dict(zip(tracks,np.array([
            self.Model.predict(
            self.Model.evolute(
                self.Q*np.random.randn(
                    self.Model.Evolution_dim, 1),
                pos),
                np.zeros((self.Model.Control_dim, 1)))
            for pos in positions])))
        out = [Measurement(1.0, self.Model.measure(pos)) for pos in self.Objects[i+1].values()]
        if self.Loose:
            for k in set(np.random.randint(low=0, high=self.N, size=int(self.N*0.1))):
                try:
                    out.pop(k)
                except IndexError:
                    pass
        return out