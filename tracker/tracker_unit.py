from abc import ABC, abstractmethod
from .unit_object import UnitObject

class TrackerUnit(ABC):
    def __init__(self):
        super().__init__()
        self.tracking_id = 0  # tracker's id
        self.unit_object = UnitObject()  # unit tracker
        self.hits = 0  # number of detection matches
        self.num_losses = 0  # number of unmatched tracks (track loss)
        self.x_state = []

    @abstractmethod
    def update(self, z):
        """
        Implement the predict and the update stages with the measurement z
        """

    @abstractmethod
    def predict(self):
        """
        Implement only the predict stage
        """
