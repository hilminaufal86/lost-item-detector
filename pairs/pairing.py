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
        all_object = self.add_xy_center(object_list)
        obj, person = self.split(np.asarray(all_object, dtype=object), self.names)

        self.obj = obj
        self.person = person
        
        for o in self.obj:
            if len(self.pair_list)==0:
                self.pair_list.append(Pair_unit(o[5], o[4], -1))
            elif not self.is_obj_in_pair_list(o[4], o[5]):
                self.pair_list.append(Pair_unit(o[5], o[4], -1))

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
        if len(self.person)==0: # no person found
            return 0            
        else:
            # check pair person for each obj
            for obj in self.obj:
                distances_person = [[self.euclidean_distance(obj[6], obj[7], p[6], p[7]), p[4]] for p in self.person]
                min_distance = min(distances_person)
                
                pair = [p for p in self.pair_list if p.is_obj_pair(obj[5], obj[4])][0]

                if pair.person_track_id == -1: # object haven't been assign to person
                    pair.person_track_id = min_distance[1]
                    pair.hit += 1

                elif pair.person_track_id != min_distance[1] and pair.hit < pair.min_hit: # person track_id diff
                    pair.person_track_id = min_distance[1]
                    pair.hit = 1
                
                elif pair.person_track_id != min_distance[1] and pair.hit >= pair.min_hit:
                    pair.diff_pair(min_distance[1])

                else: #obj[2] == min_distance[1] person track_id is same and hit >5
                    pair.same_pair()
                    pair.hit += 1
            
            # check if any person assign as warning to non exist obj
            # for per in self.person:
            #     other = [p for p in self.pair_list if p[4]==per[4]]
            #     if len(other)==0:
            #         continue

            #     is_obj_exist = False
            #     for o in other:
            #         if o[0] in self.obj[:][5] and o[1] in self.obj[:][4]:
            #             is_obj_exist = True
            #             break
                
            #     if not is_obj_exist:



    
    def reset(self, names, min_lost):
        self.names = names
        self.min_lost = min_lost
        self.pair_list = []
    
    def add_xy_center(self, obj_list):
        new_list = obj_list.copy()

        for obj in new_list:
            x_center = obj[0] + obj[2]/2
            y_center = obj[1] + obj[3]/2
            obj.extend([x_center, y_center])

        return new_list
        

    


            