class Pair_unit:
    def __init__(self, obj_class_id, obj_track_id, person_track_id, lost_limit=5, min_hit=5):
        self.obj_class_id = obj_class_id
        self.obj_track_id = obj_track_id
        self.person_track_id = person_track_id
        self.warning = 1
        self.other_track_id = -1
        self.hit = 0
        self.lost = 0
        self.lost_limit = lost_limit
        self.min_hit = min_hit

    def diff_pair(self, other_track_id):
        if self.lost > self.lost_limit:
            self.warning = 1
            self.other_track_id = other_track_id
        self.lost += 1
    
    def same_pair(self):
        self.lost = 0
        self.warning = 0
        self.other_track_id = -1
    
    def is_obj_pair(self, obj_class_id, obj_track_id):
        return self.obj_track_id==obj_track_id and self.obj_class_id==obj_class_id

    

    


