import cv2
import numpy as np

class KalmanTrack:
    _id_counter = 0

    def __init__(self, bbox):
        KalmanTrack._id_counter += 1
        self.id = KalmanTrack._id_counter
        self.hits = 1
        self.no_match = 0
        self.embedding = None

        kf = cv2.KalmanFilter(8, 4)
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.transitionMatrix[i, i + 4] = 1.0
        kf.measurementMatrix   = np.eye(4, 8, dtype=np.float32)
        kf.processNoiseCov     = np.eye(8, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        kf.errorCovPost        = np.eye(8, dtype=np.float32)

        cx, cy, w, h = self._to_state(bbox)
        kf.statePost = np.array(
            [[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=np.float32
        )
        self.kf = kf

    @staticmethod
    def _to_state(bbox):
        x1, y1, x2, y2 = bbox
        return (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1

    def predict(self):
        self.kf.predict()

    def update(self, bbox):
        cx, cy, w, h = self._to_state(bbox)
        self.kf.correct(
            np.array([[cx],[cy],[w],[h]], dtype=np.float32)
        )
        self.hits += 1
        self.no_match = 0

    def get_bbox(self):
        cx, cy, w, h = self.kf.statePost[:4].flatten()
        return int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2)