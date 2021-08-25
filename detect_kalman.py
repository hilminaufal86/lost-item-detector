import sys
import argparse
import os
import shutil
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from sort.sort import Sort
# from tracker.tracker import Tracker
# from tracker.unit_object import UnitObject
from pairs.pairing import Pairing

def plot_bbox_on_img(c1, c2, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def create_screenshot(im1, im2, save_path, obj_name, obj_id, per_id):
    label = obj_name + '-' + str(obj_id) + '-person-' + str(per_id) + '.jpg'
    path = str(Path(save_path) / Path(label))
    if os.path.exists(path):
        return
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    # create empty matrix
    img = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)

    img[:h1, :w1, :3] = im1
    img[:h2, w1:w1+w2, :3] = im2

    cv2.imwrite(path, img)
    cv2.destroyAllWindows()

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz= \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    
    if not opt.scaledyolov4:
        sys.path.insert(0, './yolov5')
        from yolov5.models.experimental import attempt_load
        from yolov5.utils.datasets import LoadStreams, LoadImages
        from yolov5.utils.general import (
            check_img_size, check_imshow, non_max_suppression, apply_classifier, scale_coords,
            xyxy2xywh)
        from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
    else:
        sys.path.insert(0, './scaledyolov4')
        from scaledyolov4.models.experimental import attempt_load
        from scaledyolov4.utils.datasets import LoadStreams, LoadImages
        from scaledyolov4.utils.general import (
            check_img_size, non_max_suppression, apply_classifier, scale_coords,
            xyxy2xywh)
        from scaledyolov4.utils.torch_utils import select_device, load_classifier, time_synchronized

    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors of object
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[0, 100, 0], [0, 204, 204]] # colors[0] - green, colors[1] - yellow (warning)

    # Initialize tracker
    # tracker = Tracker()
    tracker = Sort(max_age=20, min_hits=5, iou_threshold=0.2)

    # Initialize pairing
    pair = Pairing(names,min_lost=5)
    
    # time
    detection_total_time = 0.
    tracking_total_time = 0.
    pairing_total_time = 0.
    total_frame = 0

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    curr_vidcap = None # for different video file
    for path, img, im0s, vid_cap in dataset:
        if (curr_vidcap!=vid_cap):
            curr_vidcap = vid_cap
            # reset tracker and pair
            tracker.trackers = []
            tracker.frame_count = 0
            pair.reset(names, min_lost=5) 
        total_frame += 1         

        if webcam:
            ss_path = str(Path(out) / Path("webcam/screenshot"))
                
        else:
            ss_path = str(Path(out) / Path(path).stem / Path("screenshot"))  
        if not os.path.exists(ss_path):
            os.makedirs(ss_path)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        detection_total_time += (t2 - t1)

        # Apply Classifier (second-stage)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # initialize bboxes and coordinates
        bboxes = []
        coordinates = []

        to_sort = np.empty((0,6))
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            img_ori = im0.copy()
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                
                # Write results
                for *xyxy, conf, det_cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    # print(xyxy)
                    to_sort = np.vstack((to_sort, np.array([c1[0], c1[1], c2[0], c2[1], int(conf), int(det_cls)])))
                    # print(to_sort)
                #     if save_txt:  # Write to file
                #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #         line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                #         with open(txt_path + '.txt', 'a') as f:
                #             f.write(('%g ' * len(line) + '\n') % line)

                    # if save_img or view_img:  # Add bbox to image
                #         label = '%s %.2f' % (names[int(cls)], conf)
                        # c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                #         # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                #         # plot_bbox_on_img(c1, c2, im0, label=label, color=color[int(cls)], line_thickness=3)
                #         bboxes.append([int(cls), conf, label, c1, c2])

            t3 = time_synchronized()

            # for box in bboxes:
            #     # pindah gantiin bboxes append
            #     coordinates.append(UnitObject([box[3][0], box[3][1], box[4][0], box[4][1]], box[0]))
            
            # tracker.update(coordinates)
            tracked_dets = tracker.update(to_sort)
            # print('Output from SORT:\n',tracked_dets,'\n')
            # all_obj_list = list of (x1, y1, x2, y2, track_id, class_id)
            all_obj_list = []
            for j, track in enumerate(tracked_dets):
                x1 = track[0]
                y1 = track[1]
                x2 = track[2]
                y2 = track[3]
                track_id = track[4]
                class_id = track[5]
                # x1 = int(tracker.tracker_list[j].unit_object.xyxy[0])
                # y1 = int(tracker.tracker_list[j].unit_object.xyxy[1])
                # x2 = int(tracker.tracker_list[j].unit_object.xyxy[2])
                # y2 = int(tracker.tracker_list[j].unit_object.xyxy[3])
                # track_id = tracker.tracker_list[j].tracking_id
                # class_id = tracker.tracker_list[j].unit_object.class_id
                all_obj_list.append([x1, y1, x2, y2, track_id, class_id])
                # cv2.putText(im0, str(tracker.tracker_list[j].tracking_id), (x,y), 0, 0.5, (0, 0, 255), 2)
                # print("tracker(%d) hits: %d" % (tracker.tracker_list[j].tracking_id, tracker.tracker_list[j].hits))
                # img_label = '%s %s - %.2f' % (names[int(tracker.tracker_list[j].unit_object.class_id)], str(tracker.tracker_list[j].tracking_id), conf)
                            
            t4 = time_synchronized()

            tracking_total_time += (t4 - t3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # print('tracking time %.3fs' % (t4 - t3))

            t5 = time_synchronized()

            # # pairing
            pair.update(all_obj_list)
            pair.pair()
            
            # assign object and person to be plot in im0
            obj_list = pair.obj
            person_list = pair.person
            
            # check warning on pairing result and plot result
            for obj in obj_list:
                c1, c2 = (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])) 
                trk_id = str(obj[4])
                # print('trk_id',trk_id)
                cls_id = int(obj[5])
                # print('cls_id',cls_id)

                obj_status = [p for p in pair.pair_list if str(p.obj_track_id)==trk_id and int(p.obj_class_id)==cls_id]
                # print(obj_status)
                if len(obj_status)==0:
                    print('something wrong, the object has not been assign to pair')
                obj_status = obj_status[0]
                img_label = '%s %s' % (names[cls_id], trk_id)
                warning = obj_status.warning
                if warning:
                    img_label += ' - warning'
                plot_bbox_on_img(c1, c2, im0, label=img_label, color=colors[warning], line_thickness=1)
            
            for per in person_list:
                c1, c2 = (int(per[0]), int(per[1])), (int(per[2]), int(per[3])) 
                trk_id = str(per[4])
                cls_id = int(per[5])
                obj_status = [p for p in pair.pair_list if str(p.other_track_id)==trk_id]
                # if per[2]-per[0] <= 0 or per[3]-per[1] <=0:
                if len(obj_status)!= 0:
                    # print("Person size to be put in screenshot:")
                    # print(per[:4])
                    im_person = img_ori[c1[1]:c2[1], c1[0]:c2[0]]
                    for p in obj_status:
                        obj = [o for o in obj_list if p.obj_class_id==o[5] and p.obj_track_id==o[4]]
                        if len(obj)==0:
                            continue
                        obj = obj[0]
                        obj_name = names[int(obj[5])]
                        obj_trk_id = obj[4]
                        x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
                        # print(obj[:4])
                        # if x2-x1 <= 0 or y2-y1 <=0:
                        # print("object size to be put in screenshot:")
                        # print(obj[:4])
                            # continue
                        im_obj = img_ori[y1:y2, x1:x2]
                        create_screenshot(im_obj, im_person, ss_path, obj_name, obj_trk_id, trk_id)

                    obj_status = obj_status[0]
                    warning = obj_status.warning
                else:
                    warning = 0
                img_label = '%s %s' % (names[cls_id], trk_id)
                if warning:
                    img_label += ' - warning'
                plot_bbox_on_img(c1, c2, im0, label=img_label, color=colors[warning], line_thickness=1)
            
            t6 = time_synchronized()
            pairing_total_time += (t6 - t5)
            # print('pairing time %.3fs' % (t6 - t5))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    print('Detection Total Time (%.3fs)' % (detection_total_time))
    print('Detection Average Time (%.3fs)' % (detection_total_time/total_frame))
    print('Tracking Total Time (%.3fs)' % (tracking_total_time))
    print('Tracking Average Time (%.3fs)' % (tracking_total_time/total_frame))
    print('Pairing Total Time (%.3fs)' % (pairing_total_time))
    print('Pairing Average Time (%.3fs)' % (pairing_total_time/total_frame))
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5/weights/yolov5m_SGD.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/videos/VIRAT_S_010204.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--scaledyolov4', action='store_true', help='using yolov5 or scaledyolov4 model for object detection, default yolov5')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:
        detect()
