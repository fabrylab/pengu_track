import numpy as np
from skimage import transform
from scipy.ndimage.interpolation import map_coordinates
from skimage import img_as_uint

class HorizonEqualisation():
    # Define Warp Function
    def __init__(self, horizonmarkers, f, sensor_size, pengu_markers, h_p, max_dist, img_shape,  camera_h=None, log=False):

        self.Horizonmarkers = horizonmarkers
        self.Pengu_Markers = pengu_markers
        self.F = f
        self.Sensor_Size = sensor_size
        self.h_p = h_p
        self.Max_Dist = max_dist

        self.height, self.width  = img_shape


        x, y = self.Horizonmarkers
        m, t = np.polyfit(x, y, 1)  # linear fit for camera role
        angle_m = np.arctan(m)
        x_0 = self.width / 2
        y_0 = x_0 * m + t

        x_p1, y_p1, x_p2, y_p2 = self.Pengu_Markers
        x_p1 = (x_p1 - x_0) * np.cos(angle_m) + x_0 - (y_p1 - y_0) * np.sin(angle_m)
        y_p1 = (x_p1 - x_0) * np.sin(angle_m) + (y_p1 - y_0) * np.cos(angle_m) + y_0

        x_p2 = (x_p2 - x_0) * np.cos(angle_m) + x_0 - (y_p2 - y_0) * np.sin(angle_m)
        y_p2 = (x_p2 - x_0) * np.sin(angle_m) + (y_p2 - y_0) * np.cos(angle_m) + y_0

        # calc phi (tilt of the camera)
        self.Phi = self.calc_phi([x, y], self.Sensor_Size, [self.height, self.width], f)
        # pp = 20 * np.pi / 180.

        # get the lower ones of the markers
        args = np.argmax([y_p1, y_p2], axis=0)
        y11 = np.asarray([[y_p1[i], y_p2[i]][a] for i, a in enumerate(args)])
        x11 = np.asarray([[x_p1[i], x_p2[i]][a] for i, a in enumerate(args)])

        # calc height above plane
        tt = self.calc_theta([x11, y11], self.Sensor_Size, [self.height, self.width], f)
        gg = self.calc_gamma([x_p1, y_p1], [x_p2, y_p2], self.Sensor_Size, [self.height, self.width], f)
        hh = self.calc_height(tt, gg, self.Phi, self.h_p)

        if camera_h is None:
            self.camera_h = np.mean(hh)
            print("Height", np.mean(hh), np.std(hh) / len(hh) ** 0.5)
        else:
            self.camera_h = float(camera_h)

        # Initialize grid
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        # Grid has same aspect ratio as picture

        # counter-angle to phi is used in further calculation
        phi_ = np.pi / 2 - self.Phi

        # calculate Maximal analysed distance
        self.c = 1. / (self.camera_h / self.h_p - 1.)
        self.y_min = np.asarray(
            [0, self.camera_h * np.tan(phi_ - np.arctan(self.Sensor_Size[1] / 2. / f)), -self.camera_h])


        # calculate Penguin-Projection Size
        self.Penguin_Size = np.log(1 + self.c) * self.height / np.log(self.Max_Dist / self.y_min[1])

        # warp the grid
        self.Res = max_dist / self.height
        yy = yy * (max_dist / self.height)
        xx = (xx - self.width / 2.) * (max_dist / self.height)
        # x_s is the crossing-Point of camera mid-beam and grid plane
        self.x_s = np.asarray([0, np.tan(phi_) * self.camera_h, -self.camera_h])
        self.x_s_norm = np.linalg.norm(self.x_s)
        # vector of maximal distance to camera (in y)
        self.y_max = np.asarray([0, max_dist, -self.camera_h])
        self.y_max_norm = np.linalg.norm(self.y_max)
        self.alpha_y = np.arccos(np.dot(self.y_max.T, self.x_s).T / (self.y_max_norm * self.x_s_norm)) * -1
        if log:
            warped_xx, warped_yy = self.warp_log([xx, yy])
        else:
            warped_xx, warped_yy = self.warp_orth([xx, yy])
        # reshape grid points for image interpolation
        grid = np.asarray([warped_xx.T, warped_yy.T]).T
        self.grid = grid.T.reshape((2, self.width * self.height))
        
    def warp_orth(self, xy):
        # if True:
        #     return xy
        xx_, yy_ = xy
        # vectors of every grid point in the plane (-h)
        coord = np.asarray([xx_, yy_, -self.camera_h*np.ones_like(xx_)])
        coord_norm = np.linalg.norm(coord, axis=0)
        # calculate the angle between camera mid-beam-vector and grid-point-vector
        alpha = np.arccos(np.dot(coord.T, self.x_s).T/(coord_norm*self.x_s_norm))
        # calculate the angle between y_max-vector and grid-point-vector in the plane (projected to the plane)
        theta = np.sum((np.cross(coord.T, self.x_s) * np.cross(self.y_max, self.x_s)).T
                                 , axis=0) / (coord_norm * self.x_s_norm * np.sin(alpha) *
                                              self.y_max_norm * self.x_s_norm * np.sin(self.alpha_y))
        try:
            theta[theta>1.] = 1.
            theta[theta<-1.] = -1.
        except TypeError:
            if theta > 1.:
                theta = 1.
            elif theta < -1:
                theta = -1.
            else:
                pass
        theta = np.arccos(theta) * np.sign(xx_)
        # print(np.nanmin(theta), np.nanmax(theta))
        # from the angles it is possible to calculate the position of the focused beam on the camera-sensor
        r = np.tan(alpha)*self.F
        warp_xx = np.sin(theta)*r*self.width/self.Sensor_Size[0] + self.width/2.
        warp_yy = np.cos(theta)*r*self.height/self.Sensor_Size[1] + self.height/2.
        return warp_xx, warp_yy

    def back_warp_orth(self, xy):
        if True:
            return xy
        warp_xx, warp_yy = xy
        warp_xx = (warp_xx-self.width/2.)*self.Sensor_Size[0]/self.width
        warp_yy = (warp_yy-self.height/2.)*self.Sensor_Size[1]/self.height
        # Calculate angles in Camera-Coordinates
        theta = np.arctan2(warp_yy,
                           warp_xx)
        # theta = np.pi/2.-theta
        r = (warp_xx**2+warp_yy**2)**0.5

        # theta = np.arccos((-1) * (warp_xx-self.width/2.)*self.Sensor_Size[0]/self.width/r)

        # print(np.amin(r), np.amax(r))
        alpha = np.arctan(r/self.F)
        lam = -self.camera_h/(np.sin(theta)*np.sin(np.pi/2.-self.Phi)*np.tan(alpha)-np.cos(np.pi/2.-self.Phi))
        # print(np.amin(lam), np.amax(lam))
        # xx_ = -lam * (np.tan(alpha)*np.sin(np.pi/2.-self.Phi)*np.cos(theta)-np.sin(np.pi/2.-self.Phi))
        # yy_ = lam*np.tan(alpha)*np.sin(theta)
        xx_ = - np.tan(alpha) * lam * np.cos(theta)
        yy_ = lam * np.cos(np.pi/2. - self.Phi) * np.tan(alpha) * np.sin(theta) - lam * np.sin(np.pi/2 - self.Phi)
        return xx_, -yy_

    def calc_phi(self, horizonmarkers, sensor_size, image_size, f):
        x_h, y_h = np.asarray(horizonmarkers)

        y, x = image_size

        # linear fit and rotation to compensate incorrect camera alignment
        m, t = np.polyfit(x_h, y_h, 1)  # linear fit

        x_s, y_s = sensor_size

        # correction of the Y-axis section (after image rotation)
        t += m * x / 2

        print("Role at %s"%(np.arctan(m)*180/np.pi))

        # Calculate Camera tilt
        tilt = np.arctan((t / y - 1/2.) * (y_s / f))*-1
        print("Tilt at %s"%(tilt*180/np.pi))
        return tilt

    def calc_theta(self, theta_markers, sensor_size, image_size, f):
        x_t1, y_t1 = theta_markers
        y, x = image_size
        x_s, y_s = sensor_size
        # y_t1 = (y_t1-y/2.)*y_s/y
        # r =  (((x_t1-x/2.)*(x_s/x))**2+((y_t1-y/2.)*(y_s/y))**2)**0.5
        return np.arctan2((y_t1-y/2.)*(y_s/y), f)

    def calc_gamma(self, markers1, markers2, sensor_size, image_size, f):
        x1, y1 = markers1
        x2, y2 = markers2
        y, x = image_size
        x_s, y_s = sensor_size
        x1 = (x1-x/2.)*(x_s/x)
        y1 = (y1-y/2.)*(y_s/y)
        x2 = (x2-x/2.)*(x_s/x)
        y2 = (y2-y/2.)*(y_s/y)
        return np.arccos((x1*x2+y1*y2+f**2)/((x1**2+y1**2+f**2)*(x2**2+y2**2+f**2))**0.5)

    def calc_height(self, the, gam, p, h_t):
        alpha = np.pi/2.-p-the
        return h_t*np.abs(np.cos(alpha)*np.sin(np.pi-alpha-gam)/np.sin(gam))

    def log_to_orth(self, xy):
        # self.height / self.Sensor_Size[1] + self.height / 2.
        xx_, yy_ = xy
        xx_ -= self.width/2.
        #xx_ /= (self.Max_Dist/self.height)/(self.h_p/self.Penguin_Size)
        xx_ += self.width/2.
        yy_ = self.height - self.y_min[1]*np.exp((self.height-yy_)/self.height*np.log(self.Max_Dist/self.y_min[1]))/self.Res
        return xx_, yy_

    def orth_to_log(self, xy):
        # self.height / self.Sensor_Size[1] + self.height / 2.
        xx_, yy_ = xy
        xx_ -= self.width/2.
        #xx_ /= (self.h_p/self.Penguin_Size)/(self.Max_Dist/self.height)
        xx_ += self.width/2.
        yy_ = self.height - np.log((self.height - yy_)*(self.Res/self.y_min[1]))*(self.height/np.log(self.Max_Dist/self.y_min[1]))
        return xx_, yy_

    def horizon_equalisation(self, image):
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

        if len(image.shape) == 3:
            # split the image in colors and perform interpolation
            return np.asarray([map_coordinates(i, self.grid[:, ::-1], order=0).reshape((self.width, self.height))[::-1] for i in image.T]).T
        elif len(image.shape) == 2:
            return np.asarray(img_as_uint(map_coordinates(image.T, self.grid[:, ::-1], order=0)).reshape((self.width, self.height))[::-1]).T
        else:
            raise ValueError("The given image is whether RGB nor greyscale!")