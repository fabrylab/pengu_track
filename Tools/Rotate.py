from __future__ import print_function
import glob
import piexif

rot2exif = {0: 0, 90: 6, 180: 3, 270: 8, -90: 8, -180: 3, -270: 6}

def ChangeImageRotation(filename, rotation):
    if rotation in rot2exif:
        rotation = rot2exif[rotation]
    exif_dict = piexif.load(filename)
    if piexif.ImageIFD.Orientation in exif_dict["0th"]:
        print(filename+":", "Orientation changed from", exif_dict["0th"][piexif.ImageIFD.Orientation], "to", rotation)
        exif_dict["0th"][piexif.ImageIFD.Orientation] = rotation
        piexif.insert(piexif.dump(exif_dict), filename)

images = glob.glob("*.JPG")
for im in images:
    ChangeImageRotation(im, 180)
