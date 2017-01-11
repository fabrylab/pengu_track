from __future__ import division, print_function
import numpy as np
from skimage import transform
from scipy.ndimage.interpolation import map_coordinates
from skimage.measure import regionprops, label
import sys
from skimage import img_as_uint

def horizontal_equalisation(image, horizonmarkers, f, sensor_size, h=1, markers=[], max_dist=None):
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
    Phi = np.arctan((t * (2 / Y) - 1.) * (Y_s / 2. / f))*-1
    print("Phi at %s" % (Phi * 180 / np.pi))

    # Initialize grid
    xx, yy = np.meshgrid(np.arange(X), np.arange(Y))
    # Grid has same aspect ratio as picture, but width max_dist
    yy = yy * (max_dist/Y)
    xx = (xx-X/2.) * (max_dist/Y)
    # counter-angle to phi is used in further calculation
    phi_ = np.pi/2 - Phi
    # x_s is the crossing-Point of camera mid-beam and grid plane
    x_s = np.asarray([0, np.tan(phi_)*h, -h])
    x_s_norm = np.linalg.norm(x_s)
    # vector of maximal distance to camera (in y)
    y_max = np.asarray([0, max_dist, -h])
    y_max_norm = np.linalg.norm(y_max)
    alpha_y = np.arccos(np.dot(y_max.T, x_s).T/(y_max_norm*x_s_norm)) * -1

    # Define Warp Function
    def warp(xy):
        xx_, yy_ = xy
        # vectors of every grid point in the plane (-h)
        coord = np.asarray([xx_, yy_, -h*np.ones_like(xx_)])
        coord_norm = np.linalg.norm(coord, axis=0)
        # calculate the angle between camera mid-beam-vector and grid-point-vector
        alpha = np.arccos(np.dot(coord.T, x_s).T/(coord_norm*x_s_norm)) #* np.sign(np.tan(phi_)*h-yy_)
        # calculate the angle between y_max-vector and grid-point-vector in the plane (projected to the plane)
        theta = np.arccos(np.sum((np.cross(coord.T, x_s) * np.cross(y_max, x_s)).T
                       ,axis=0)/(coord_norm*x_s_norm*np.sin(alpha) *
                                  y_max_norm*x_s_norm*np.sin(alpha_y))) * np.sign(xx) #* np.sign(np.tan(phi_)*h-yy_)
        # from the angles it is possible to calculate the position of the focused beam on the camera-sensor
        r = np.tan(alpha)*f
        warp_xx = np.sin(theta)*r*X/X_s + X/2.
        warp_yy = np.cos(theta)*r*Y/Y_s + Y/2.
        return warp_xx, warp_yy

    def back_warp(xy):
        warp_xx, warp_yy = np.asarray(xy)
        # calculate angles in the coordinate system of the sensor
        r = np.sqrt(((warp_xx - X/2.) * (X_s/X))**2 + ((warp_yy - Y/2.) * (Y_s/Y))**2)
        a1 = np.arctan2((warp_xx - X/2.) * (X_s/X), f)
        a2 = np.arctan2((warp_yy - Y/2.) * (Y_s/Y), f)
        print(np.array([a1,a2]).T*180/np.pi)
        x = np.tan(a1)*x_s_norm
        yy = (np.sin(phi_)*x_s_norm+x_s_norm*np.sin(a2)/np.sin(np.pi/2-a2-phi_))
        xx = yy*x/(x_s_norm*np.sin(phi_))
        print(np.array([xx,yy]).T)

        alpha = np.arctan2(r, f)
        theta = np.arctan2(((warp_xx - X/2.) * (X_s/X)), ((warp_yy - Y/2.) * (Y_s/Y)))

        # # plt.scatter(theta*180/np.pi, alpha*180/np.pi)
        # # plt.show()
        e_1 = np.cross(y_max, x_s)*(1/(y_max_norm*x_s_norm*np.sin(alpha_y)))
        e_2 = np.cross(e_1, x_s)*(1/x_s_norm)
        e_s = x_s*(1/x_s_norm)

        x = x_s + (np.tan(alpha)*x_s_norm * (np.tensordot(np.cos(theta), e_2, axes=0)+np.tensordot(np.sin(theta), e_1, axes=0)).T).T
        e_x= x.T/np.linalg.norm(x.T, axis=0)
        print(e_x)
        print(np.linalg.norm(e_x, axis=0))
        lam = h/e_x[2]
        print(lam)
        xx, yy, zz = lam*e_x
        return xx*Y/max_dist+X/2, yy*Y/max_dist+Y
        # print(e_1,e_2,e_s)
        #
        # lam_s = (y_max_norm*x_s_norm)/(x_s_norm*np.cos(theta)*np.tan(alpha)
        #                                - y_max_norm*np.cos(alpha_y)*np.cos(theta)*np.tan(alpha)
        #                                + y_max_norm)
        # # lam_s = x_s_norm * np.ones_like(lam_s)
        # lam_1 = np.sin(theta)*np.tan(alpha)*lam_s
        # lam_2 = np.cos(theta)*np.tan(alpha)*lam_s
        # # print (lam_s,lam_1,lam_2)
        # x = np.tensordot(lam_1, e_1, axes=0) + np.tensordot(lam_2,e_2, axes=0) + np.tensordot(lam_s,e_s, axes=0)
        # xx, yy, zz = np.asarray(x).T
        # return X/2-xx*Y/max_dist, Y-yy*Y/max_dist

    # warp the grid
    warped_xx, warped_yy = warp([xx,yy])
    # reshape grid points for image interpolation
    grid = np.asarray([warped_xx.T, warped_yy.T]).T
    grid = grid.T.reshape((2, X * Y))

    # warp the markers

    if len(markers) == 0:
        if len(image.shape) == 3:
            # split the image in colors and perform interpolation
            return np.asarray([img_as_uint(map_coordinates(i, grid[:, ::-1])).reshape((X, Y))[::-1] for i in image.T]).T
        elif len(image.shape) == 2:
            return np.asarray(img_as_uint(map_coordinates(image.T, grid[:, ::-1])).reshape((X, Y))[::-1]).T
        else:
            raise ValueError("The given image is whether RGB nor greyscale!")
    else:
        # if markers are given, also equalise them, with the same function
        # new_markers = np.array([back_warp(m) for m in markers])
        new_markers = back_warp(markers.T)
        # split the image in colors and perform interpolation#
        if len(image.shape) == 3:
            # split the image in colors and perform interpolation
            return np.asarray([img_as_uint(map_coordinates(i, grid[:, ::-1])).reshape((X, Y))[::-1] for i in image.T]).T,\
               np.asarray(new_markers).T
        elif len(image.shape) == 2:
            return np.asarray(img_as_uint(map_coordinates(image.T, grid[:, ::-1])).reshape((X, Y))[::-1]).T,\
               np.asarray(new_markers).T
        else:
            raise ValueError("The given image is whether RGB nor greyscale!")

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
h = float(input("Please enter Height above projected plane in m: \n"))
f = float(input("Please enter Focal length in mm: \n")) * 1e-3
sizex, sizey = input("Please enter Sensor Size (X,Y) in mm: \n")
SensorSize = np.asarray([sizex, sizey], dtype=float) * 1e-3
max_dist = input("Please enter maximal analysed distance to camera in m: \n")
Y, X = image.getShape()
marker_type = db.setMarkerType(name="Projection-Correction Area marker", color="#FFFFFF")

for mask in db.getMasks(frame=frame):
    data = mask.data
    img = mask.image
    labeled_org = label(data)
    region_pos = {}
    for props in regionprops(labeled_org):
        region_pos.update({props.label: props.centroid})
    n_regions = np.amax(region_pos.keys())

    eq = horizontal_equalisation(labeled_org, [x, y], f, SensorSize,
                                              h=h,
                                              max_dist=max_dist)


    # eq2 = horizontal_equalisation(image.data, [x, y], f, SensorSize,
    #                                           h=h,
    #                                           max_dist=max_dist)
    #
    # # if input("Show image? (Y/N) \n").count("Y"):
    # import matplotlib.pyplot as plt
    # plt.imshow(labeled_org)
    # plt.figure()
    # plt.imshow(image.data)
    # mi = eq.min()
    # ma = eq.max()
    # eq = ((eq - mi)*n_regions/(ma-mi)+0.5).astype(int)-1 #haha mami
    # eq = img_as_uint(eq)
    # plt.figure()
    # plt.imshow(eq)
    # plt.figure()
    # plt.imshow(eq2)
    # plt.show()

    # eq = eq.astype(int)
    for prop in regionprops(eq):
        i = prop.label
        try:
            db.setMarker(frame=frame, x=region_pos[i][1], y=region_pos[i][0], text="area %s"%prop.area, type=marker_type)
        except KeyError:
            pass

    com.ReloadMarker(image.sort_index)
    # check if we should terminate
    if com.HasTerminateSignal():
        print("Cancelled Correction")
        sys.exit(0)

#
# if __name__ == "__main__":
#     import clickpoints
#     import matplotlib.pyplot as plt
#
#     # Load database
#     db = clickpoints.DataFile("Horizon.cdb")
#
#     #Camera parameters
#     f = 14e-3  # Focal length
#     SensorSize = (17.3e-3, 13.e-3)
#     start_image = db.getImage(frame=0).data[::-1, ::-1]  # raw image is rotated
#     Y, X = start_image.shape[:2]  # get image size
#
#     # Load horizon-markers, rotate them
#     horizont_type = db.getMarkerType(name="Horizon")
#     x, y = np.array([[X-m.x, Y-m.y] for m in db.getMarkers(type=horizont_type)]).T
#     # x, y = np.array([[m.x, m.y] for m in db.getMarkers(type=horizont_type)]).T
#
#     # Load Position markers, rotate them
#     position_type = db.getMarkerType(name="Position_old")
#     markers = np.array([[X-m.x, Y-m.y] for m in db.getMarkers(type=position_type)]).astype(int)
#     # markers = np.array([[m.x, m.y] for m in db.getMarkers(type=position_type)]).astype(int)
#     # do correction
#     image, markers2 = horizontal_equalisation(start_image, [x, y], f, SensorSize,
#                                               h=25,
#                                               markers=markers,
#                                               max_dist=Y/10)
#
#     # plot this shit
#     plt.imshow(image)
#     plt.scatter(markers2.T[0], markers2.T[1])
#     plt.show()
