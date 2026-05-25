import cv2
import queue
import threading
from ultralytics import YOLO
from tracker import MOTracker
from reid_encoder import ReIDEncoder
from visualizer import draw_tracks

# ── Threaded video reader ──────────────────────────────────────────
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=4)
        self.stopped = False
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            if not self.q.full():
                self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        self.cap.release()

# ── Main loop ──────────────────────────────────────────────────────
def run(source=0, conf=0.4, classes=[0]):
    """
    source : 0 for webcam, or path to video file e.g. 'video.mp4'
    conf   : detection confidence threshold
    classes: YOLO class IDs to track  (0 = person)
    """
    detector = YOLO("yolov8n.pt")        # downloads automatically on first run
    encoder  = ReIDEncoder(device="cpu") # change to "cuda" if you have a GPU
    tracker  = MOTracker(max_age=30, min_hits=3)
    stream   = VideoStream(source)

    while not stream.stopped:
        frame = stream.read()

        results = detector(frame, conf=conf,
                           classes=classes, verbose=False)

        bboxes = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                bboxes.append((x1,y1,x2,y2))

        embeddings = encoder.encode(frame, bboxes)
        active     = tracker.update(bboxes, embeddings)
        frame      = draw_tracks(frame, active)

        cv2.putText(frame, f"Tracks: {len(active)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,255,255), 2)
        cv2.imshow("Multi Object Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run(source=0)   # swap 0 for "your_video.mp4" to use a file