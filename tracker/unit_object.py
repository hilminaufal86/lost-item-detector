from typing import List

class UnitObject:
    def __init__(self, xyxy=[], idx=None):
        """
        Create bounding box with id of vector
        :param bounds: [left, top, right, bottom] -> [x1, y1, x2, y2]
        :param id: type of classification
        """
        self.xyxy: List[int] = xyxy
        self.class_id: int = idx

    def __str__(self):
        return str(self.class_id) + ": " + str(self.xyxy)