from __future__ import division, print_function
import numpy as np
from skimage import transform


def horizontal_equalisation(image, horizonmarkers, f, sensor_size, h=1, markers=[]):
    """
    Parameters:
    ----------
    image: array-like
        Image.
    horizonmarkers:
        List of at least 3 points at horizon in the image.
    f: float
        Focal length of the objective.
    sensor_size: array-like
        Sensor-size of the used camera.
    h: float, optional
    Height of the camera over the shown plane.
    markers: array-like, optional
        List of markers in the image, which should also be transformed

    Returns:
    ----------
    image: array-like
        Equalised image.
    new_markers: array-like
        Equalised markers.
    """
    x_h, y_h = np.asarray(horizonmarkers)
    # Rotate image
    image = np.asarray(image)[::-1, ::-1]
    Y, X = image.shape[:2]
    # Rotate horizon markers
    x_h = X - x_h
    y_h = Y - y_h
    # Rotate other markers
    if np.any(markers):
        markers = np.array(markers)
        markers.T[0] = X - markers.T[0]
        markers.T[1] = Y - markers.T[1]

    # linear fit and rotation to compensate incorrect camera alignment
    m, t = np.polyfit(x_h, y_h, 1)  # linear fit
    image = np.array(image, ndmin=2, dtype=np.uint8)
    image = transform.rotate(image, np.arctan(m) * 180 / np.pi)

    X_s, Y_s = sensor_size

    # correction of the Y-axis section (after image rotation)
    t -= m * X / 2

    # Calculate Camera tilt
    Phi = np.arctan((t * (2 / Y) - 1.) * (Y_s / 2. / f))
    print("Phi at %s" % (Phi * 180 / np.pi))

    # define function for the skimage-warp-correction
    def true_y(xy):
        xx, yy = np.asarray(xy).T

        # remove points beyond t (horizon)
        yy = (yy + t) * (Y / (Y + t))
        # Calculate y-angle for each pixel
        warp_yy = np.arctan((yy * (2. / Y) - 1.) * (Y_s / 2. / f))
        # Compensate for tilt and equalise picture (this is where the magic happens)
        warp_yy = h*np.tan(np.pi / 2 - Phi - warp_yy)

        # Rescale data to fit to grid (i do not know, why i need this, but it works)
        warp_min = h*np.tan(np.pi / 2 - Phi - np.arctan(Y_s / 2. / f))
        warp_max = h*np.tan(np.pi / 2 - Phi - np.arctan((t/(Y+t)*2-1) * Y_s / 2. / f)) - warp_min

        warp_yy -= warp_min
        warp_yy /= warp_max
        warp_yy = (1 - warp_yy) * t

        # x-transformation to regain aspect ratio (and tilt penguins to the side...)
        warp_xx = (xx - X / 2.) / (warp_yy / yy) + X / 2.
        return np.asarray([warp_xx.T, warp_yy.T]).T

        # do the correction with transform.warp
    if len(markers) == 0:
        return transform.warp(image, true_y, clip=False, preserve_range=True)[::-1, ::-1]
    else:
        # if markers are given, also equalise them, with the same function
        new_markers = np.array([true_y(m) for m in markers])
        new_markers.T[0] = X - new_markers.T[0]
        new_markers.T[1] = Y - new_markers.T[1]
        return transform.warp(image, true_y)[::-1, ::-1], new_markers


if __name__ == "__main__":
    import clickpoints
    import matplotlib.pyplot as plt

    # Load database
    db = clickpoints.DataFile("horizon2.cdb")

    #Camera parameters
    f = 14e-3  # Focal length
    SensorSize = (17.3e-3, 13.e-3)
    start_image = db.getImage(frame=0).data[::-1, ::-1] # raw image is rotated
    Y, X = start_image.shape[:2] # get image size

    # Load horizon-markers, rotate them
    horizont_type = db.getMarkerType(name="horizon")
    # x, y = np.array([[X-m.x, Y-m.y] for m in db.getMarkers(type=horizont_type)]).T
    x, y = np.array([[m.x, m.y] for m in db.getMarkers(type=horizont_type)]).T

    # Load Position markers, rotate them
    position_type = db.getMarkerType(name="Position")
    # markers = np.array([[X-m.x, Y-m.y] for m in db.getMarkers(type=position_type)]).astype(int)
    markers = np.array([[m.x, m.y] for m in db.getMarkers(type=position_type)]).astype(int)
    # do correction
    image, new_markers = horizontal_equalisation(start_image, [x, y], f, SensorSize, h=1,  markers=markers)

    # plot this shit
    plt.imshow(image)
    # plt.scatter(new_markers.T[0], new_markers.T[1])
    plt.show()
