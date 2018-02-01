# Tracker (assignment and data handling)
from PenguTrack.Filters import MultiFilter
# Kalman Filter (storage of data, prediction, representation of tracks)
from PenguTrack.Filters import KalmanFilter
# Standard Modules for parametrization
import scipy.stats as ss
import numpy as np

# Physical Model (used for predictions)
from PenguTrack.Models import VariableSpeed


def VariableSpeedTracker(dim=2, object_size=1., q=1., r=1.):
    log_prob_threshold = -20.  # Threshold for track stopping

    # Initialize physical model as 2d variable speed model with 0.5 Hz frame-rate
    model = VariableSpeed(dim=dim, timeconst=1)

    # Set up Kalman filter
    X = np.zeros(model.State_dim).T  # Initial Value for Position
    Q = np.diag([q * object_size * np.ones(model.Evolution_dim)])  # Prediction uncertainty
    R = np.diag([r * object_size * np.ones(model.Meas_dim)])  # Measurement uncertainty
    State_Dist = ss.multivariate_normal(cov=Q)  # Initialize Distributions for Filter
    Meas_Dist = ss.multivariate_normal(cov=R)  # Initialize Distributions for Filter
    # Initialize Filter/Tracker
    MultiKal = MultiFilter(KalmanFilter, model, np.diag(Q),
                           np.diag(R), meas_dist=Meas_Dist, state_dist=State_Dist, no_dist=True)
    MultiKal.LogProbabilityThreshold = log_prob_threshold

    return MultiKal