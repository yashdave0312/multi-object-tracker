import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from kalman_track import KalmanTrack

def iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def build_cost_matrix(tracks, detections, det_embeddings,
                      iou_w=0.6, app_w=0.4, iou_thresh=0.2):
    n, m = len(tracks), len(detections)
    C = np.ones((n, m))
    for i, tr in enumerate(tracks):
        tb = tr.get_bbox()
        for j, det in enumerate(detections):
            iou_score = iou(tb, det)
            if iou_score < iou_thresh:
                continue
            iou_cost = 1 - iou_score
            app_cost = 0.0
            if tr.embedding is not None and det_embeddings[j] is not None:
                app_cost = cosine(tr.embedding, det_embeddings[j])
            C[i, j] = iou_w * iou_cost + app_w * app_cost
    return C

class MOTracker:
    def __init__(self, max_age=30, min_hits=3):
        self.tracks: list[KalmanTrack] = []
        self.max_age = max_age
        self.min_hits = min_hits

    def update(self, detections, det_embeddings):
        for tr in self.tracks:
            tr.predict()
            tr.no_match += 1

        if not detections:
            self.tracks = [t for t in self.tracks
                           if t.no_match <= self.max_age]
            return self._active()

        C = build_cost_matrix(self.tracks, detections, det_embeddings)
        row_ind, col_ind = linear_sum_assignment(C)

        matched_tracks, matched_dets = set(), set()
        for r, c in zip(row_ind, col_ind):
            if C[r, c] < 0.85:
                self.tracks[r].update(detections[c])
                if det_embeddings[c] is not None:
                    self.tracks[r].embedding = det_embeddings[c]
                matched_tracks.add(r)
                matched_dets.add(c)

        for j, det in enumerate(detections):
            if j not in matched_dets:
                t = KalmanTrack(det)
                t.embedding = det_embeddings[j]
                self.tracks.append(t)

        self.tracks = [t for t in self.tracks
                       if t.no_match <= self.max_age]
        return self._active()

    def _active(self):
        return [t for t in self.tracks if t.hits >= self.min_hits]