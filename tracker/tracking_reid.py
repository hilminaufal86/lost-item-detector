import numpy as np
import torch
import cv2
import time
from .detection import Detection
from .tracker import Tracker
from .nn_matching import NearestNeighborDistanceMetric


import sys
# sys.path.append('./fast_reid/demo')
sys.path.append('./fast_reid')
# from demo import Reid_feature
from fastreid.engine.defaults import DefaultPredictor

__all__ = ['TrackingReid']

def _xyxy_to_tlwh(bbox_xyxy):
    # print(bbox_xyxy)
    if isinstance(bbox_xyxy, np.ndarray):
        bbox_tlwh = bbox_xyxy.copy()
    elif isinstance(bbox_xyxy, torch.Tensor):
        bbox_tlwh = bbox_xyxy.clone()
    bbox_tlwh[:][2] = bbox_xyxy[:][0] + bbox_xyxy[:][2]
    bbox_tlwh[:][3] = bbox_xyxy[:][1] + bbox_xyxy[:][3]
    return bbox_tlwh

class TrackingReid(object):
    def __init__(self, cfg, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.extractor = DefaultPredictor(cfg)
        self.dist_metric = NearestNeighborDistanceMetric('euclidean', max_dist, nn_budget)
        self.tracker = Tracker(self.dist_metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def reset(self):
        self.tracker = Tracker(self.dist_metric, max_iou_distance=self.max_iou_distance, max_age=self.max_age, n_init=self.n_init)

    def update(self, bbox_xyxy, confidences, ori_img):
        outputs = []
        if not len(confidences):
            return outputs
        else:
            self.height, self.width = ori_img.shape[:2]
            xyxy = np.array(bbox_xyxy,dtype=object)
            print('\nxyxy:',xyxy[:,0])
            features = self._get_features(xyxy[:,0], ori_img)
            bbox_tlwh = _xyxy_to_tlwh(np.asarray(xyxy[:,0]))
            print('\ntlwh:',bbox_tlwh)
            cls_id = xyxy[:,1]
            # bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
            
            detections = [Detection(cls_id,bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences)] # if conf>self.min_confidence]

            # run non-maximum supression
            # boxes = np.array([d.tlwh for d in detections])
            # scores = np.array([d.confidence for d in detections])
            # indices = non_max_supression(boxes, self.nms_max_overlap, scores)
            # detections = [detections[i] for i in indices]

            # update tracker
            self.tracker.predict()
            self.tracker.update(detections)

            # output bbox
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                # track_id = track.track_id
                outputs.append(np.array([x1, y1, x2, y2, track.track_id, track.class_id], dtype=np.int))
            
            if len(outputs) > 0:
                outputs = np.stack(outputs, axis=0)
            
            np.asarray(outputs, dtype=object)
            np.asarray(features, dtype=object)
            return outputs, features

    # @staticmethod
    # def _xywh_to_tlwh(bbox_xywh):
    #     if isintance(bbox_xywh, np.ndarray):
    #         bbox_tlwh = bbox_xywh.copy()
    #     elif isintance(bbox_xywh, torch.Tensor):
    #         bbox_tlwh = bbox_xywh.clone()
    #     bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
    #     bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
    #     return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2), 0)
        y1 = max(int(y-h/2), 0)
        x2 = min(int(x+w/2), self.width-1)
        y2 = min(int(y+h/2), self.height-1)
        return x1, y1, x2, y2
    
    def _tlwh_to_xyxy(self, bbox_tlwh):
        x,y,w,h = bbox_tlwh
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x+w), self.width-1)
        y2 = min(int(y+h), self.height-1)

        return x1, y1, x2, y2

    # def _xyxy_to_tlwh(self, bbox_xyxy):
    #     x1,y1,x2,y2 = bbox_xyxy
    #     w = int(x2-x1)
    #     h = int(y2-y1)
    #     return x1, y1, w, h

    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        t1 = time.time()

        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            # print(ori_img)
            # print(box,self._xywh_to_xyxy(box))
            # print('box: ',box)
            im = ori_img[int(y1):int(y2), int(x1):int(x2)]
            # print('image: ',im)
            im = im[:, :, ::-1] # reid pretreatment
            # print('image: ',im)
            im = cv2.resize(im, (128, 256), interpolation=cv2.INTER_CUBIC)
            im_crops.append(torch.as_tensor(im.astype('float32').transpose(2, 0, 1))[None])
        print('pre-time:',time.time() - t1)

        if im_crops:
            t1 = time.time()
            batch_image = torch.cat(im_crops, dim=0)
            features = self.extractor(batch_image)
            print('reid features extraction time:', time.time()-t1,' - ', features.shape)
        else:
            features = np.array([])
        
        return features
            

