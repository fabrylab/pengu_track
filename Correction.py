from __future__ import division, print_function
import numpy as np
from skimage import transform


def horizontal_equalisation(image, horizonmarkers, f, sensor_size):

    x_h, y_h = np.asarray(horizonmarkers)
    m, t = np.polyfit(x_h, y_h, 1) #  linear fit
    image = np.array(image, ndmin=2, dtype=np.uint8)
    image = transform.rotate(image, np.arctan(m)*180/np.pi)

    Y, X = image.shape[:2]

    X_s, Y_s = sensor_size

    Phi = (t/Y-0.5) * 2 * np.arctan(Y_s/2./f)  #Calculate Camera tilt
    print("Phi at %s"%(Phi*180/np.pi))

    def true_y(xy):
        xx, yy = np.asarray(xy).T
        y_max = yy.max()
        # remove points beyond t (horizon)
        warp_yy = (yy+t)*(y_max/(y_max+t))
        # Calculate y-angle for each pixel
        alpha_yy = (warp_yy/Y-0.5) * 2 * np.arctan(Y_s/2./f)
        # Compensate for tilt and equalise picture (this is where the magic happens)
        warp_yy = np.tan(np.pi/2-Phi-alpha_yy)
        # Rescale data to fit to grid
        warp_yy -= warp_yy.min()
        warp_yy /= warp_yy.max()
        warp_yy = (1-warp_yy)*t
        return np.asarray([xx.T, warp_yy.T]).T

    return transform.warp(image, true_y)

if __name__ == "__main__":
    import clickpoints
    db = clickpoints.DataFile("./Developement/adelie_data/770_PANA/Horizont.cdb")

    x, y = np.array([[m.x, m.y] for m in db.getMarkers(frame=0)]).T  # Load horizon points
    f = 14e-3  # Focal length
    SensorSize = (17.3e-3, 13.e-3)
    image = horizontal_equalisation(db.getImage(frame=0).data, [x, y], f, SensorSize)
