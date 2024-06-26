import ctypes


class Point:
    X: ctypes.c_double
    Y: ctypes.c_double
    Z: ctypes.c_double
    R: ctypes.c_uint16
    G: ctypes.c_uint16
    B: ctypes.c_uint16
    intensity: ctypes.c_uint8
    pointId: ctypes.c_uint32
    isEmpty: ctypes.c_bool

    def __init__(self, X, Y, Z, R, G, B, intensity, pointId, isEmpty) -> None:
        self.X = X
        self.Y = Y
        self.Z = Z
        self.R = R
        self.G = G
        self.B = B
        self.pointId = pointId
        self.intensity = intensity
        self.isEmpty = isEmpty

    def __repr__(self) -> str:
        return f"{self.pointId}, ({self.X}, {self.Y}, {self.Z}), ({self.R}, {self.G}, {self.B}), {self.intensity}, {self.isEmpty}"
