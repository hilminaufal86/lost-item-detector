import numpy as np
import math
from .pair_unit import Pair_unit

class Pairing:
    '''
    all_object_list = [x1, y1, x2, y2, track_id, class_id] => object and person not separate yet
    obj, person = [x1, y1, x2, y2, track_id, class_id] => [x1, y1, x2, y2, track_id, class_id, x center, y center]
    pair_list = list of (obj_class_id, obj_track_id, person_track_id, warning(0 or 1), other_person_track_id/-1, lost_time)
    '''
    def __init__(self, names, min_lost=5):
        self.names = names
        self.min_lost = min_lost
        self.pair_list = []

    def update(self, object_list):
        if isinstance(object_list, np.ndarray):
            print("object list already instance of np")
        # self.all_obj_list = object_list
        all_object = self.add_xy_center(object_list)
        obj, person = self.split(np.asarray(all_object, dtype=object), self.names)
        # self.obj = self.add_xy_center(obj)
        # self.person = self.add_xy_center(person)
        self.obj = obj
        self.person = person
        for o in self.obj:
            if len(self.pair_list)==0:
                self.pair_list.append(Pair_unit(o[5], o[4], -1))
            # elif o[4] not in self.pair_list[:].obj_track_id and o[5] not in self.pair_list[:].obj_class_id:
                # self.pair_list.append([o[5], o[4], -1, 1, -1, 0])
            elif not self.is_obj_in_pair_list(o[4], o[5]):
                self.pair_list.append(Pair_unit(o[5], o[4], -1))
        # self.pair_list = [[obj_cls_id, obj_trk_id, -1, 0, -1, 0] for _, _, _, _, obj_trk_id, obj_cls_id in self.obj]

    def is_obj_in_pair_list(self, obj_track_id, obj_class_id):
        for p in self.pair_list:
            if p.obj_track_id==obj_track_id and p.obj_class_id==obj_class_id:
                return True
        return False

    def split(self, object_list, names):
        obj = []
        person = []
        for x in object_list:
            
            if names[int(x[5])]=='person':
                person.append(x)
            else:
                obj.append(x)

        obj = np.asarray(obj)
        person = np.asarray(person)

        return obj, person
    
    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((math.pow(x1-x2,2)) + math.pow(y1-y2,2))
    
    def pair(self):
        # pair_list = self.pair_list
        if len(self.person)==0: # no person found
            return 0            
        else:
            for obj in self.obj:
                distances_person = [[self.euclidean_distance(obj[6], obj[7], x, y), trk_id] for _, _, _, _, trk_id, _, x, y in self.person]
                min_distance = min(distances_person)
                
                pair = [p for p in self.pair_list if p.is_obj_pair(obj[5], obj[4])][0]

                if pair.person_track_id == -1: # object haven't been assign to person
                    pair.person_track_id = min_distance[1]

                if pair.person_track_id != min_distance[1]: # person track_id diff
                    pair.diff_pair(min_distance[1])

                else: #obj[2] == min_distance[1] person track_id is same
                    pair.same_pair()

        # return self.pair_list
    
    def reset(self, names, min_lost):
        self.names = names
        self.min_lost = min_lost
        self.pair_list = []
    
    def add_xy_center(self, obj_list):
        new_list = obj_list.copy()
    #     half_point = [x[2] / 2 for x in obj_list]
    #     x_center = [obj_list[:,0] + half_point]
    #     half_point = [x[3] / 2 for x in obj_list]
    #     y_center = [obj_list[:,1] + half_point]
    #     print('xcenter:',x_center)

        # new_list = np.append(new_list, x_center, axis=0)
        # new_list = np.append(new_list, y_center, axis=0)
        for obj in new_list:
            x_center = obj[0] + obj[2]/2
            y_center = obj[1] + obj[3]/2
            # obj = np.append(obj, [x_center, y_center], axis=0)
            # obj.extend([x_center, y_center])
        print(new_list)
        return new_list


    # def output_xywh_to_xyxy(self, width, height):
    #     obj_xyxy = self.obj.copy()
    #     if len(self.obj)!=0:
    #         half_width = [x[2] / 2 for x in self.obj]
    #         half_height = [x[3] / 2 for x in self.obj]
    #         obj_xyxy[:,0] = [max(0, int(obj[0]-obj[2]/2)) for obj in self.obj]
    #         obj_xyxy[:,0] = [max(0, int(obj[1]-obj[3]/2)) for obj in self.obj]
    #         obj_xyxy[:,0] = [min(width-1, int(obj[0]+obj[2]/2)) for obj in self.obj]
    #         obj_xyxy[:,0] = [min(height-1, int(obj[1]+obj[3]/2)) for obj in self.obj]
    #     # print(obj_xyxy[:])
    #     # obj_xyxy[:,0] = max(0, int(self.obj[:,0]-half_width))
    #     # obj_xyxy[:,1] = max(0, int(self.obj[:,1]-half_height))
    #     # obj_xyxy[:,2] = min(int(self.obj[:,0]+half_width), width-1)
    #     # obj_xyxy[:,3] = min(int(self.obj[:,1]+half_height), height-1)

    #     person_xyxy = self.person.copy()
    #     if len(self.person)!=0:
    #         half_width = [x[2] / 2 for x in self.person]
    #         half_height = [x[3] / 2 for x in self.person]
    #         person_xyxy[:,0] = [max(0, int(per[0]-per[2]/2)) for per in self.person]
    #         person_xyxy[:,0] = [max(0, int(per[1]-per[3]/2)) for per in self.person]
    #         person_xyxy[:,0] = [max(width-1, int(per[0]+per[2]/2)) for per in self.person]
    #         person_xyxy[:,0] = [max(height-1, int(per[1]+per[3]/2)) for per in self.person]
    #     # person_xyxy[:,0] = max(0, int(self.person[:,0]-half_width))
    #     # person_xyxy[:,1] = max(0, int(self.person[:,1]-half_height))
    #     # person_xyxy[:,2] = min(int(self.person[:,0]+half_width), width-1)
    #     # person_xyxy[:,3] = min(int(self.person[:,1]+half_height), height-1)

    #     return obj_xyxy, person_xyxy



        

    


            