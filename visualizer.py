import cv2
from collections import defaultdict

PALETTE = [
    (255,56,56),  (255,157,151),(255,112,31), (255,178,29),
    (207,210,49), (72,249,10),  (146,204,23), (61,219,134),
    (26,147,52),  (0,212,187),  (44,153,168), (0,194,255),
    (52,69,147),  (100,115,255),(0,24,236),   (132,56,255),
    (82,0,133),   (203,56,255), (255,149,200),(255,55,199)
]

trails = defaultdict(list)

def draw_tracks(frame, active_tracks, trail_len=40):
    for tr in active_tracks:
        x1, y1, x2, y2 = tr.get_bbox()
        cx, cy = (x1+x2)//2, (y1+y2)//2
        color = PALETTE[tr.id % len(PALETTE)]

        trails[tr.id].append((cx, cy))
        if len(trails[tr.id]) > trail_len:
            trails[tr.id].pop(0)

        pts = trails[tr.id]
        for k in range(1, len(pts)):
            alpha = k / len(pts)
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, pts[k-1], pts[k], color, thickness)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        label = f"ID {tr.id}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(frame, (x1, y1-th-8),
                      (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255,255,255), 1)
    return frame