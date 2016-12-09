from __future__ import division, print_function
import numpy as np
from skimage import transform


def horizontal_equalisation(image, horizonmarkers, f, sensor_size):

    x_h, y_h = np.asarray(horizonmarkers)
    image = image[::-1, ::-1]
    Y, X = image.shape[:2]
    x_h = X-x_h
    y_h = Y-y_h
    m, t = np.polyfit(x_h, y_h, 1) #  linear fit
    image = np.array(image, ndmin=2, dtype=np.uint8)
    image = transform.rotate(image, np.arctan(m)*180/np.pi)

    X_s, Y_s = sensor_size

    t -= m*X/2

    Phi = np.arctan((t*(2/Y)-1.)*(Y_s/2./f))  #Calculate Camera tilt
    print("Phi at %s"%(Phi*180/np.pi))

    def true_y(xy):
        xx, yy = np.asarray(xy).T

        y_max = yy.max()
        # remove points beyond t (horizon)
        yy = (yy+t)*(y_max/(y_max+t))
        # Calculate y-angle for each pixel
        warp_yy = np.arctan((yy*(2./Y)-1.)*(Y_s/2./f))
        # Compensate for tilt and equalise picture (this is where the magic happens)
        warp_yy = np.tan(np.pi/2-Phi-warp_yy)
        # Rescale data to fit to grid
        warp_yy -= warp_yy.min()
        warp_yy /= warp_yy.max()
        warp_yy = (1-warp_yy)*t
        #x-transformation to regain aspect ratio
        warp_xx = (xx-X/2.)/(warp_yy/yy)+X/2.
        return np.asarray([warp_xx.T, warp_yy.T]).T

    return transform.warp(image, true_y)[::-1, ::-1]

if __name__ == "__main__":
    import clickpoints
    db = clickpoints.DataFile("testbild.cdb")

    horizont_type = db.getMarkerType(name="Horizont")
    x, y = np.array([[m.x, m.y] for m in db.getMarkers(type=horizont_type)]).T  # Load horizon points
    f = 14e-3  # Focal length
    SensorSize = (17.3e-3, 13.e-3)
    start_image = db.getImage(frame=0).data
    Y, X = start_image.shape[:2]
    image = horizontal_equalisation(start_image, [x, y], f, SensorSize)
