import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag

from .tracker_unit import TrackerUnit

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form [x,y,s,r] where x,y is the center of the box and s is the scale/area and r is the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_z_to_bbox(z):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(z[2] * z[3])
    h = z[2] / w

    return np.array([z[0]-w/2.,z[1]-h/2.,z[0]+w/2.,z[1]+h/2.])

class KalmanTracker(TrackerUnit):
    """
    class for Kalman Filter-based tracker
    """

    def __init__(self, bbox):
        super().__init__()
        # Initialize parameters for Kalman Filtering
        # The state is the (x, y) coordinates of the detection box
        # state: [up, up_dot, left, left_dot, down, down_dot, right, right_dot]
        # or[up, up_dot, left, left_dot, height, height_dot, width, width_dot]
        # self.dt = 1.  # time interval
        # # self.x_state = np.zeros((4, 1))

        # # Process matrix, assuming constant velocity model
        # self.F = np.array([[1, self.dt, 0, 0, 0, 0, 0, 0],
        #                    [0, 1, 0, 0, 0, 0, 0, 0],
        #                    [0, 0, 1, self.dt, 0, 0, 0, 0],
        #                    [0, 0, 0, 1, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 1, self.dt, 0, 0],
        #                    [0, 0, 0, 0, 0, 1, 0, 0],
        #                    [0, 0, 0, 0, 0, 0, 1, self.dt],
        #                    [0, 0, 0, 0, 0, 0, 0, 1]])

        # # Measurement matrix, assuming we can only measure the coordinates
        # self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
        #                    [0, 0, 1, 0, 0, 0, 0, 0],
        #                    [0, 0, 0, 0, 1, 0, 0, 0],
        #                    [0, 0, 0, 0, 0, 0, 1, 0]])

        # # Initialize the state covariance
        # self.L = 100.0
        # self.P = np.diag(self.L * np.ones(8))

        # # Initialize the process covariance
        # self.Q_comp_mat = np.array([[self.dt ** 4 / 4., self.dt ** 3 / 2.],
        #                             [self.dt ** 3 / 2., self.dt ** 2]])
        # self.Q = block_diag(self.Q_comp_mat, self.Q_comp_mat,
        #                     self.Q_comp_mat, self.Q_comp_mat)

        # # Initialize the measurement covariance
        # self.R_scaler = 10.0
        # self.R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
        # self.R = np.diag(self.R_diag_array)
        dim_x = 7
        dim_z = 4
        self.x = np.zeros((dim_x, 1))
        self.x[:4] = convert_bbox_to_z(bbox)

        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        # self.B = None
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        # self._alpha_sq = 1.
        # self.M = np.zeros((dim_x, dim_z))
        self.z = np.array([[None]*dim_z]).T

        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))
        # self.SI = np.zeros((dim_z, dim_z))
        
        # identity matrix
        self._I = np.eye(dim_x)

        # copy of x,P after predict()
        # self.x_prior = self.x.copy()
        # self.P_prior = self.P.copy()

        # copy of x,P after update()
        # self.x_post = self.x.copy()
        # self.P_post = self.P.copy()

        # self._mahalanobis = None
        # self.inv = np.linalg.inv

        self.setup()

    # def update_R(self):
    #     R_diag_array = self.R_scaler * np.array([self.L, self.L, self.L, self.L])
    #     self.R = np.diag(R_diag_array)
    def setup(self):
        self.F = np.array([[1,0,0,0,1,0,0],
                           [0,1,0,0,0,1,0],
                           [0,0,1,0,0,0,1],
                           [0,0,0,1,0,0,0],
                           [0,0,0,0,1,0,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,1]])

        self.H = np.array([[1,0,0,0,0,0,0],
                           [0,1,0,0,0,0,0],
                           [0,0,1,0,0,0,0],
                           [0,0,0,1,0,0,0]])

        self.R[2:, 2:] *= 10. # R: Covariance matrix of measurement noise (set to high for noisy inputs -> more 'inertia' of boxes')
        self.P[4:, 4:] *= 100. # give high uncertainty to the unobservable initial velocities
        self.P *= 10.
        self.Q[-1, -1] *= 0.5 #Q: Covariance matrix of process noise (set to high for erratically moving things)
        self.Q[4:, 4:] *= 0.5

    def predict(self):
        # x = self.x_state
        # Predict
        self.x = dot(self.F, self.x) # predicted state estimate
        self.P = dot(self.F, self.P).dot(self.F.T) + self.Q # predicted error covariance 
        # self.x_state = x.astype(int)

    def update(self, z):
        # x = self.x_state
        # # Predict
        # x = dot(self.F, x) # predicted state estimate
        # self.P = dot(self.F, self.P).dot(self.F.T) + self.Q # predicted error covariance 

        # Update
        # y = np.array(z.copy()).reshape(4,1)
        # z = self.x.copy()
        # print(y.shape)
        # print(z.shape)
        # print(z)
        # z[:4] = convert_bbox_to_z(y)
        # print(dot(self.H, self.x).shape)
        z = np.array(convert_bbox_to_z(z)).reshape(4,1)
        self.S = self.R + dot(self.H, self.P).dot(self.H.T)
        self.y = z - dot(self.H, self.x)  # residual
        self.K = dot(self.P, self.H.T).dot(inv(self.S))  # Kalman gain
        self.x += dot(self.K, self.y) # update state estimate
        # self.P = self.P - dot(K, self.H).dot(self.P) # update error covariance
        I_KH = self._I - dot(self.K, self.H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, self.R), self.K.T)
        # self.x_state = x.astype(int)  # convert to integer coordinates

    def get_x_bbox(self):
        return convert_z_to_bbox(self.x)
