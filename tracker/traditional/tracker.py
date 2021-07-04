import logging
from collections import deque
from typing import List

import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment


from .unit_object import UnitObject
from .base_tracker import BaseTracker
from .kalman_tracker import KalmanTracker

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.WARN)


def calculate_iou(box1, box2):
    """
    Calculate intersection over union
    :param box1: a[0], a[1], a[2], a[3] <-> left, top, right, bottom
    :param box2: b[0], b[1], b[2], b[3] <-> left, top, right, bottom
    """

    w_intersec = np.maximum(0, (np.minimum(box1[2], box2[2]) - np.maximum(box1[0], box2[0])))
    h_intersec = np.maximum(0, (np.minimum(box1[3], box2[3]) - np.maximum(box1[1], box2[1])))
    area_intersec = w_intersec * h_intersec

    area_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_b = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return float(area_intersec) / (area_a + area_b - area_intersec)

class Tracker:
    """
    Class that connects detection and tracking
    """

    def __init__(self, minHits = 3, maxAge = 20):
        self.max_age = maxAge
        self.min_hits = minHits
        self.tracker_list: List[BaseTracker] = []
        # self.track_id_list = deque(list(map(str, range(25))))
        self.track_id = 1
        self.tracker = KalmanTracker()

    def update(self, unit_detections):


        unit_trackers = []

        for trk in self.tracker_list:
            unit_trackers.append(trk.unit_object)

        matched, unmatched_dets, unmatched_trks = self.assign_detections_to_trackers(unit_trackers, unit_detections,
                                                                                     iou_thrd=0.3)

        # print('Detection: ' + str(unit_detections))
        # print('x_box: ' + str(unit_trackers))
        # print('matched:' + str(matched))
        # print('unmatched_det:' + str(unmatched_dets))
        # print('unmatched_trks:' + str(unmatched_trks))

        # Matched Detections
        for track_idx, det_idx in matched:
            z = unit_detections[det_idx].xyxy
            z = np.expand_dims(z, axis=0).T
            temp_track = self.tracker_list[track_idx]
            temp_track.predict_and_update(z)
            xx = temp_track.x_state.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            unit_trackers[track_idx].xyxy = xx
            unit_trackers[track_idx].class_id = unit_detections[det_idx].class_id
            temp_track.unit_object = unit_trackers[track_idx]
            temp_track.hits += 1
            temp_track.num_losses = 0

        # Unmatched Detections -> detected object, unmatched track -> set as new track
        for idx in unmatched_dets:
            z = unit_detections[idx].xyxy
            z = np.expand_dims(z, axis=0).T
            temp_track = KalmanTracker() #self.tracker()  # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            temp_track.x_state = x
            temp_track.predict_only()
            xx = temp_track.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            temp_track.unit_object.xyxy = xx
            temp_track.unit_object.class_id = unit_detections[idx].class_id
            # temp_track.tracking_id = self.track_id_list.popleft()  # assign an ID for the tracker
            temp_track.tracking_id = self.track_id
            self.track_id += 1
            self.tracker_list.append(temp_track)
            unit_trackers.append(temp_track.unit_object)

        # Unmatched trackers -> missing object
        for track_idx in unmatched_trks:
            temp_track = self.tracker_list[track_idx]
            temp_track.num_losses += 1
            temp_track.predict_only()
            xx = temp_track.x_state
            xx = xx.T[0].tolist()
            xx = [xx[0], xx[2], xx[4], xx[6]]
            temp_track.unit_object.box = xx
            unit_trackers[track_idx] = temp_track.unit_object

        # The list of tracks to be annotated
        # good_tracker_list = []
        # for track in self.tracker_list:
        #     if (track.hits >= self.min_hits) and (track.num_losses <= self.max_age):
        #         good_tracker_list.append(track)
                # img = utils.drawing.draw_box_label(img, trk, self.detector.class_names)

        # Manage Tracks to be deleted
        # deleted_tracks = filter(lambda x: x.num_losses > self.max_age, self.tracker_list)

        # for trk in deleted_tracks:
        #     self.track_id_list.append(trk.tracking_id)

        # remove track with age > max age
        self.tracker_list = [x for x in self.tracker_list if x.num_losses <= self.max_age]

    def reset(self, minHits = 3, maxAge = 20 ):
        self.max_age = maxAge
        self.min_hits = minHits
        self.tracker_list: List[BaseTracker] = []
        self.track_id = 1
        self.tracker = KalmanTracker()
    

    @staticmethod
    def assign_detections_to_trackers(unit_trackers: List[UnitObject], unit_detections: List[UnitObject], iou_thrd=0.3):
        """
        Matches Trackers and Detections
        :param unit_trackers: trackers
        :param unit_detections: detections
        :param iou_thrd: threshold to qualify as a match
        :return: matches, unmatched_detections, unmatched_trackers
        """
        IOU_mat = np.zeros((len(unit_trackers), len(unit_detections)), dtype=np.float32)
        for t, trk in enumerate(unit_trackers):
            for d, det in enumerate(unit_detections):
                if trk.class_id == det.class_id:
                    IOU_mat[t, d] = calculate_iou(trk.xyxy, det.xyxy)

        # Finding Matches using Hungarian Algorithm
        row_ind, col_ind = linear_assignment(-IOU_mat)
        # matched_idx = linear_assignment(-IOU_mat)

        unmatched_trackers, unmatched_detections = [], []
        for t, trk in enumerate(unit_trackers):
            if t not in row_ind: # matched_idx[:, 0]:
                unmatched_trackers.append(t)

        for d, det in enumerate(unit_detections):
            if d not in col_ind: #matched_idx[:, 1]:
                unmatched_detections.append(d)

        matches = []

        # Checking quality of matched by comparing with threshold
        for i in range(len(row_ind)):
            if IOU_mat[row_ind[i], col_ind[i]] < iou_thrd:
                unmatched_trackers.append(row_ind[i])
                unmatched_detections.append(col_ind[i])
            else:
                m = np.array([row_ind[i], col_ind[i]])
                matches.append([m.reshape(1, 2)])

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)


        matches = matches.reshape(len(matches), 2)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
