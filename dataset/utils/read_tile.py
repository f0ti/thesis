from fileinput import filename
from tile import Tile

filename = "pc2img/Tile_+003_+005_0_1"
tile = Tile(filename)
tile.save()
tile.show()
