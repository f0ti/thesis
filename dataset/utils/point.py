import ctypes

class Point:
    def __init__(self, X, Y, Z, R, G, B, intensity, pointId, isEmpty) -> None:
        self.X: ctypes.c_double = X
        self.Y: ctypes.c_double = Y
        self.Z: ctypes.c_double = Z
        self.R: ctypes.c_uint16 = R
        self.G: ctypes.c_uint16 = G
        self.B: ctypes.c_uint16 = B
        self.pointId: ctypes.c_uint8 = pointId
        self.intensity: ctypes.c_uint32 = intensity
        self.isEmpty: ctypes.c_bool = isEmpty

    def __repr__(self) -> str:
        return f"{self.pointId}, ({self.X}, {self.Y}, {self.Z}), ({self.R}, {self.G}, {self.B}), {self.intensity}, {self.isEmpty}"
