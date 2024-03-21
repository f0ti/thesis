import torch

class Point:
    X: torch.half  # 16-bit signed floating point
    Y: torch.half
    Z: torch.half
    R: torch.uint8  # unsigned 8-bit integer
    G: torch.uint8
    B: torch.uint8
    intensity: torch.half

    def __init__(self, X, Y, Z, R, G, B, intensity) -> None:
        self.X = X
        self.Y = Y
        self.Z = Z
        self.R = R
        self.G = G
        self.B = B
        self.intensity = intensity

    def __repr__(self) -> str:
        return f"({self.X}, {self.Y}, {self.Z}), ({self.R}, {self.G}, {self.B}), {self.intensity}"

    @property
    def X(self):
        return self.X

    @property
    def Y(self):
        return self.Y

    @property
    def Z(self):
        return self.Z

    @property
    def R(self):
        return self.R

    @property
    def G(self):
        return self.G

    @property
    def B(self):
        return self.B

    @property
    def intensity(self):
        return self.intensity
