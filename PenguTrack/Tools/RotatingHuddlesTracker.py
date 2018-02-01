import numpy as np
import cv2
from PenguTrack.Detectors import FlowDetector, EmperorDetector, rgb2gray
from PenguTrack.Filters import HungarianTracker, KalmanFilter
from PenguTrack.Models import BallisticWSpeed
from PenguTrack.DataFileExtended import DataFileExtended
import matplotlib.pyplot as plt
import scipy.stats as ss
db = DataFileExtended(r"D:\User\Alex\Documents\Promotion\RotatingHuddles\DataBases\data1980.cdb")
db.deletetOld()

# FD = FlowDetector()

object_size = 3  # Object diameter (smallest)
q = 1.  # Variability of object speed relative to object size
r = 1.  # Error of object detection relative to object size
log_prob_threshold = -20.  # Threshold for track stopping

model = BallisticWSpeed(dim=2)

# Set up Kalman filter
X = np.zeros(model.State_dim).T  # Initial Value for Position
# Q = np.diag([q * object_size * np.ones(model.Evolution_dim)])  # Prediction uncertainty
Q = np.diag([q * object_size * np.array([0.1, 1.0, 0.1, 1.0])])  # Prediction uncertainty

# R = np.diag([r * object_size * np.ones(model.Meas_dim)])  # Measurement uncertainty
R = np.diag([r * object_size * np.array([0.1, 1.0, 0.1, 1.0])])  # Measurement uncertainty

State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter

MultiKal = HungarianTracker(KalmanFilter, model, np.diag(Q),
                            np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist,
                            measured_variables=["PositionX", "VelocityX", "PositionY", "VelocityY"])
MultiKal.LogProbabilityThreshold = log_prob_threshold

images = db.getImageIterator(start_frame=220)
# images = db.getImageIterator(start_frame=0, end_frame=220)
image = next(images)
# image_data = image.data#[200:400, 0:200]
# image_int = rgb2gray(image_data)
# FD.update(np.ones_like(image_int, dtype=bool), image_int)
i=0
db_model, db_tracker = db.init_tracker(model, MultiKal)
ED = EmperorDetector(image.data, luminance_threshold=1.8)
for image in images:
    MultiKal.predict(i=i)
    # image_data = image.data#[200:400,0:200]
    # image_int = rgb2gray(image_data)
    # flow = FD.segmentate(image_int)
    # FD.update(np.ones_like(image_int, dtype=bool), image_int)
    measurements = ED.detect(image.data)#[200:300, 750:1000])
    # print(len(measurements))
    # for m in measurements:
    #     m.PositionX += 200
    #     m.PositionY += 750
    MultiKal.update(z=measurements, i=i)
    db.write_to_DB(MultiKal, image, i=i, text="", db_model=db_model, db_tracker=db_tracker)
    i+=1
    print(i)
    break


"""
from mpl_toolkits.axes_grid1 import make_axes_locatable

# images = db.getImageIterator(start_frame=35, end_frame=50)
images = db.getImageIterator(start_frame=0, end_frame=220)
image = next(images)
# image.data = image.data[210:270, 100:500]
image_int = rgb2gray(image.data)
FD.update(np.ones_like(image_int, dtype=bool), image_int)
for image in images:
    print(image.sort_index)
    image_data = np.array(image.data)
    # image.data = image.data[210:270,100:500]
    image_int = rgb2gray(image.data)
    flow = FD.segmentate(image_int)
    FD.update(np.ones_like(image_int, dtype=bool), image_int)

    fig, ax1 = plt.subplots()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax1.axis('tight')
    ax1.axis('off')
    ax1.imshow(image_data)
    import matplotlib as mpl
    flowX = flow.T[0].T
    flowY = flow.T[1].T

    from skimage.filters import gaussian
    amax = np.amax(np.abs(flow))
    flowX = gaussian(flowX/amax, sigma=15)*amax
    flowY = gaussian(flowY/amax, sigma=15)*amax
    flow.T[0]=flowX.T
    flow.T[1]=flowY.T

    flow_dir = np.arctan2(flow.T[1], flow.T[0]).T
    flow_int = np.linalg.norm(flow, axis=-1)
    cmap = mpl.cm.get_cmap('hsv')

    flow_mapped = cmap(flow_dir)
    flow_mapped.T[3] = flow_int.T/np.amax(flow_int)

    ax1.imshow(image_data)
    # ax1.imshow(flow_mapped)

    # from skimage.feature import peak_local_max
    # peaksX, peaksY = peak_local_max(image_int,min_distance=5, threshold_abs=30).T
    from PenguTrack.Detectors import RegionPropDetector
    from PenguTrack.Detectors import RegionFilter
    rf1 = RegionFilter("area", 75, lower_limit= 30, upper_limit=200)
    # # rf2 = RegionFilter("in_out_contrast2", 75)#, lower_limit= 30, upper_limit=300)
    area_detector = RegionPropDetector([rf1])
    measurements = area_detector.detect(image_int>30,image_int)
    peaksX = np.array([m.PositionX for m in measurements])
    peaksY = np.array([m.PositionY for m in measurements])
    # ax1.plot(peaksY, peaksX, "rx")

    from scipy.interpolate import RegularGridInterpolator

    interpolatorX = RegularGridInterpolator((np.arange(flowX.shape[0]), np.arange(flowX.shape[1])), flowX)
    interpolatorY = RegularGridInterpolator((np.arange(flowY.shape[0]), np.arange(flowY.shape[1])), flowY)

    peaks = np.array([peaksX, peaksY]).T[0]
    peakflowX = np.array(interpolatorX(peaks))#-image.offset.y
    peakflowX -=np.median(peakflowX)
    peakflowY = np.array(interpolatorY(peaks))#-image.offset.x
    peakflowY -=np.median(peakflowY)
    peakDir = np.arctan2(peakflowY, peakflowX)

    map = cmap((peakDir+np.pi)/(2*np.pi))
    for x,y,dx,dy,c in zip(peaksY.T[0],peaksX.T[0], peakflowX, peakflowY,map):
        ax1.arrow(x,y,dx*10,dy*10, color=c)
        # ax1.arrow(x+100,y+210,dx,dy, color=c)
    # ax1.plot([100,100,500,500,100],[210,270,270,210,210], "r-")
    ax1.set_ylim([540,0])
    ax1.set_xlim([0,960])

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.55)
    cax.imshow(np.arange(0,1,0.01)[:,None], extent=[0,20,-180,180], cmap="hsv")
    cax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    # cax.yaxis.tick_right()

    # hist_x, bins_x = np.histogram(peaksX, bins=18, normed=True)
    # hist_x -= np.amin(hist_x)
    # hist_y, bins_y = np.histogram(peaksY, bins=40, normed=True)
    # hist_y -= np.amin(hist_y)
    # plt.plot(hist_x*1e4, 270-bins_x[1:], "g-")
    # plt.plot(500-bins_y[1:], 540-hist_y*1e4, "g-")
    plt.savefig(r"F:\analysis\rotating_huddles\data3_m31n3_20140703-0500_crop\FlowArrows%s.png"%str(image.sort_index).zfill(4))
"""
