import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag

from .base_tracker import BaseTracker

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}
    
class KalmanTracker(BaseTracker):
    """
    class for Kalman Filter-based tracker
    """

    def __init__(self):
        super().__init__()
        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        self.dt = 1.  # time interval

        # Process matrix, assuming constant velocity model
        self.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, self.dt, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.dt, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, self.dt],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        # Measurement matrix, assuming we can only measure the coordinates
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0]])

        # Initialize the state covariance
        self.L = 10.0
        self.P = np.diag(self.L * np.ones(8))

        # Initialize the process covariance
        self.Q_comp_mat = np.array([[self.dt ** 4 / 4., self.dt ** 3 / 2.],
                                    [self.dt ** 3 / 2., self.dt ** 2]])
        self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
                            self.Q_comp_mat, self.Q_comp_mat)

        # Initialize the measurement covariance
        self.R_scaler = 1.0
        self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(self.R_diag_array)

    def update_R(self):
        R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        self.R = np.diag(R_diag_array)

    def predict_and_update(self, z):
        x = self.x_state
        # Predict
        x = dot(self.F, x) # predicted state estimate
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q # predicted error covariance 

        # Update
        S = self.R + dot(self.H, self.P).dot(self.H.T)
        y = z - dot(self.H, x)  # residual
        K = dot(self.P, self.H.T).dot(inv(S))  # Kalman gain
        x += dot(K, y) # update state estimate
        self.P = self.P - dot(K, self.H).dot(self.P) # update error covariance
        self.x_state = x.astype(int)  # convert to integer coordinates

    def predict_only(self):
        x = self.x_state
        # Predict
        x = dot(self.F, x) # predicted state estimate
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q # predicted error covariance 
        self.x_state = x.astype(int)
