from __future__ import division, print_function
import numpy as np
from skimage import transform
from scipy.ndimage.interpolation import map_coordinates
from skimage.measure import regionprops, label
import sys
from skimage import img_as_uint


def horizontal_equalisation(image, horizonmarkers, f, sensor_size, h_p, h=1, markers=[], max_dist=None):
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
    max_dist: float, optional
        Maximum distance (distance from camera into direction of horizon) to be shown.

    Returns:
    ----------
    image: array-like
        Equalised image.
    new_markers: array-like
        Equalised markers.
    """
    if not max_dist:
        max_dist = np.sqrt(2*6371e3*h+h**2)
    x_h, y_h = np.asarray(horizonmarkers)
    # Rotate image
    image = np.asarray(image)#[::-1, ::-1]

    Y, X = image.shape[:2]

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
    t += m * X / 2
    print(t)

    # Calculate Camera tilt
    Phi = phi([x_h, y_h], [X_s, Y_s], [Y, X], f)
    print("Tilt at %s" % (Phi * 180 / np.pi))
    # Initialize grid
    xx, yy = np.meshgrid(np.arange(X), np.arange(Y))
    # Grid has same aspect ratio as picture, but width max_dist
    yy = yy * (max_dist/Y)
    xx = (xx-X/2.) * (max_dist/Y)#*3./4.
    # counter-angle to phi is used in further calculation
    phi_ = np.pi/2 - Phi
    # x_s is the crossing-Point of camera mid-beam and grid plane
    x_s = np.asarray([0, np.tan(phi_)*h, -h])
    x_s_norm = np.linalg.norm(x_s)
    # vector of maximal distance to camera (in y)
    y_max = np.asarray([0, max_dist, -h])
    y_max_norm = np.linalg.norm(y_max)
    # y_min = np.asarray([0, h*np.tan(phi_-np.arctan(Y_s/f)), -h])
    alpha_y = np.arccos(np.dot(y_max.T, x_s).T/(y_max_norm*x_s_norm)) * -1

    # Define Warp Function
    def warp(xy):
        xx_, yy_ = xy
        # vectors of every grid point in the plane (-h)
        coord = np.asarray([xx_, yy_, -h*np.ones_like(xx_)])
        coord_norm = np.linalg.norm(coord, axis=0)
        # calculate the angle between camera mid-beam-vector and grid-point-vector
        alpha = np.arccos(np.dot(coord.T, x_s).T/(coord_norm*x_s_norm))
        # calculate the angle between y_max-vector and grid-point-vector in the plane (projected to the plane)
        theta = np.sum((np.cross(coord.T, x_s) * np.cross(y_max, x_s)).T
                                 , axis=0) / (coord_norm * x_s_norm * np.sin(alpha) *
                                              y_max_norm * x_s_norm * np.sin(alpha_y))
        theta[theta>1.] = 1.
        theta[theta<-1.] = -1.
        theta = np.arccos(theta) * np.sign(xx_)
        # print(np.nanmin(theta), np.nanmax(theta))
        # from the angles it is possible to calculate the position of the focused beam on the camera-sensor
        r = np.tan(alpha)*f
        warp_xx = np.sin(theta)*r*Y/Y_s + X/2.
        warp_yy = np.cos(theta)*r*Y/Y_s + Y/2.
        return warp_xx, warp_yy

    # Define Warp Function
    def warp2(xy):
        xx_, yy_ = xy
        old = np.array(yy_)
        # vectors of every grid point in the plane (-h)
        # print(max_dist/Y)
        c = 1./(h/h_p-1.)
        yy_ /= max_dist # 0 bis 1
        yy_ *= np.log(max_dist/y_min[1])
        # print(np.amin(yy_), np.amax(yy_))
        # 1 - > Y*c
        # yy_ = (yy_) * ((Y*c-1)/max_dist) + 1 # 1 -> Y*c
        # yy_ = np.cumsum(1./(yy_+1), axis=0) # 1 -> 1/(Y*c)

        yy_ = y_min[1]*np.exp(yy_)
        p_s = np.log(1+c)/(np.log(max_dist/y_min[1])/Y)
        # print("Penguin size is %s pixels!"%p_s)
        # mmm = np.amax(yy_)
        # yy_ *= (max_dist/mmm)
        # print(np.amin(yy_), np.amax(yy_))
        # import matplotlib.pyplot as plt
        # plt.scatter(old[:, 0], yy_[:, 0])
        # plt.show()
        print(xx_.shape, yy_.shape)
        coord = np.asarray([xx_, yy_, -h*np.ones_like(xx_)])
        coord_norm = np.linalg.norm(coord, axis=0)
        # calculate the angle between camera mid-beam-vector and grid-point-vector
        alpha = np.arccos(np.dot(coord.T, x_s).T/(coord_norm*x_s_norm)) #* np.sign(np.tan(phi_)*h-yy_)
        # calculate the angle between y_max-vector and grid-point-vector in the plane (projected to the plane)
        print(coord.shape, x_s.shape, y_max.shape)
        theta = np.arccos(np.sum((np.cross(coord.T, x_s) * np.cross(y_max, x_s)).T
                       ,axis=0)/(coord_norm*x_s_norm*np.sin(alpha) *
                                  y_max_norm*x_s_norm*np.sin(alpha_y))) * np.sign(xx_) #* np.sign(np.tan(phi_)*h-yy_)
        # from the angles it is possible to calculate the position of the focused beam on the camera-sensor
        r = np.tan(alpha)*f
        warp_xx = np.sin(theta)*r*X/X_s + X/2.
        warp_yy = np.cos(theta)*r*Y/Y_s + Y/2.
        return warp_xx, warp_yy

    warped_xx, warped_yy = warp([xx, yy])
    # reshape grid points for image interpolation
    grid = np.asarray([warped_xx.T, warped_yy.T]).T
    grid = grid.T.reshape((2, X * Y))

    if len(image.shape) == 3:
        # split the image in colors and perform interpolation
        return np.asarray([map_coordinates(i, grid[:, ::-1], order=0).reshape((X, Y))[::-1] for i in image.T]).T
    elif len(image.shape) == 2:
        return np.asarray(img_as_uint(map_coordinates(image.T, grid[:, ::-1], order=0)).reshape((X, Y))[::-1]).T
    else:
        raise ValueError("The given image is whether RGB nor greyscale!")


def phi(horizonmarkers, sensor_size, image_size, f):
    x_h, y_h = np.asarray(horizonmarkers)

    y, x = image_size

    # linear fit and rotation to compensate incorrect camera alignment
    m, t = np.polyfit(x_h, y_h, 1)  # linear fit

    x_s, y_s = sensor_size

    # correction of the Y-axis section (after image rotation)
    t += m * x / 2

    print("Role at %s"%(np.arctan(m)*180/np.pi))

    # Calculate Camera tilt
    return np.arctan((t / y - 1/2.) * (y_s / f))*-1


def theta(theta_markers, sensor_size, image_size, f):
    x_t1, y_t1 = theta_markers
    y, x = image_size
    x_s, y_s = sensor_size
    # y_t1 = (y_t1-y/2.)*y_s/y
    # r =  (((x_t1-x/2.)*(x_s/x))**2+((y_t1-y/2.)*(y_s/y))**2)**0.5
    return np.arctan2((y_t1-y/2.)*(y_s/y), f)


def gamma(markers1, markers2, sensor_size, image_size, f):
    x1, y1 = markers1
    x2, y2 = markers2
    y, x = image_size
    x_s, y_s = sensor_size
    x1 = (x1-x/2.)*(x_s/x)
    y1 = (y1-y/2.)*(y_s/y)
    x2 = (x2-x/2.)*(x_s/x)
    y2 = (y2-y/2.)*(y_s/y)
    return np.arccos((x1*x2+y1*y2+f**2)/((x1**2+y1**2+f**2)*(x2**2+y2**2+f**2))**0.5)


def height(the, gam, p, h_t):
    alpha = np.pi/2.-p-the
    return h_t*np.abs(np.cos(alpha)*np.sin(np.pi-alpha-gam)/np.sin(gam))

if __name__ == "__main__":
    import clickpoints
    # Connect to database
    frame, database, port = clickpoints.GetCommandLineArgs()
    db = clickpoints.DataFile(database)
    com = clickpoints.Commands(port, catch_terminate_signal=True)

    # get the images
    image = db.getImage(frame=frame)

    # Load horizon-markers, rotate them
    horizont_type = db.getMarkerType(name="Horizon")
    x, y = np.array([[m.x, m.y] for m in db.getMarkers(type=horizont_type)]).T

    Y, X = image.getShape()
    # h = float(input("Please enter Height above projected plane in m: \n"))
    f = float(input("Please enter Focal length in mm: \n")) * 1e-3
    sizex, sizey = input("Please enter Sensor Size (X,Y) in mm: \n")
    SensorSize = np.asarray([sizex, sizey], dtype=float) * 1e-3
    print(Y)
    max_dist = input("Please enter maximal analysed distance to camera in m: \n")
    res = max_dist/Y
    marker_type = db.setMarkerType(name="Projection-Correction Area marker", color="#FFFFFF")
    com.ReloadTypes()



    # Load penguin-markers, rotate them
    penguin_type = db.getMarkerType(name="Penguin_Size")
    x_p1, y_p1, x_p2, y_p2 = np.array([[m.x1, m.y1, m.x2, m.y2] for m in db.getLines(type="Penguin_Size")]).T

    # Actual penguin size in m
    h_p = float(penguin_type.text) #0.525#0.6
    print("Size at ", h_p)

    m, t = np.polyfit(x, y, 1)  # linear fit for camera role
    angle_m = np.arctan(m)
    x_0 = X/2
    y_0 = x_0*m+t

    x_p1 = (x_p1-x_0) * np.cos(angle_m) + x_0 - (y_p1-y_0) * np.sin(angle_m)
    y_p1 = (x_p1-x_0) * np.sin(angle_m) + (y_p1-y_0) * np.cos(angle_m) + y_0

    x_p2 = (x_p2-x_0) * np.cos(angle_m) + x_0 - (y_p2-y_0) * np.sin(angle_m)
    y_p2 = (x_p2-x_0) * np.sin(angle_m) + (y_p2-y_0) * np.cos(angle_m) + y_0

    # calc phi (tilt of the camera)
    pp = phi([x, y], SensorSize, [Y, X], f)
    # pp = 20 * np.pi / 180.

    # get the lower ones of the markers
    args = np.argmax([y_p1, y_p2], axis=0)
    y11 = np.asarray([[y_p1[i], y_p2[i]][a] for i, a in enumerate(args)])
    x11 = np.asarray([[x_p1[i], x_p2[i]][a] for i, a in enumerate(args)])

    # calc height above plane
    tt = theta([x11, y11], SensorSize, [Y, X], f)
    gg = gamma([x_p1, y_p1], [x_p2, y_p2], SensorSize, [Y, X], f)
    hh = height(tt, gg, pp, h_p)

    print("Height", np.mean(hh), np.std(hh)/len(hh)**0.5)
    h = np.mean(hh)

    for mask in db.getMasks(frame=frame):
        data = mask.data
        img = mask.image
        labeled_org = label(data)
        region_pos = {}
        for props in regionprops(labeled_org):
            region_pos.update({props.label: props.centroid})
        n_regions = np.amax(region_pos.keys())

        eq = horizontal_equalisation(labeled_org, [x, y], f, SensorSize, h_p,
                                                  h=h,
                                                  max_dist=max_dist)
        eq2 = horizontal_equalisation(image.data, [x, y], f, SensorSize, h_p,
                                                  h=h,
                                                  max_dist=max_dist)
        import matplotlib.pyplot as plt
        plt.imshow(eq2)
        plt.show()

        eq = ((eq / np.amax(eq)) * n_regions).astype(int)
        print("%s old regions"%n_regions)
        for prop in regionprops(eq):
            i = prop.label
            print(i, prop.area)
            try:
                db.setMarker(frame=frame, x=region_pos[i][1], y=region_pos[i][0], text="label %s area %s"%(i, float(prop.area*res**2)), type=marker_type)
            except KeyError:
                pass

        com.ReloadMarker(image.sort_index)
        # check if we should terminate
        if com.HasTerminateSignal():
            print("Cancelled Correction")
            sys.exit(0)
