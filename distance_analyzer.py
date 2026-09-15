import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from ultralytics import YOLO

img_source = '/Users/vyankov/Documents/Datasets/cat-fights/task1/1.mov'
model_path = 'best.pt'
min_thresh = 0.5

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Running inference on device: {device}')

def get_centroids(detections):
    order = np.argsort(-detections.conf.cpu().numpy())
    centroids = []
    for i in order:
        conf = detections[i].conf.item()
        if conf <= min_thresh:
            continue
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        centroids.append(np.array([cx, cy]))
    return centroids

def analyze_video(cap, model):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    distances = []
    times = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break

        results = model(frame, conf=min_thresh, device=device, verbose=False)
        centroids = get_centroids(results[0].boxes)

        if len(centroids) == 2:
            dist = np.linalg.norm(centroids[0] - centroids[1])
            distances.append(dist)
        else:
            distances.append(np.nan)

        times.append(frame_count / fps)
        frame_count += 1
    return distances, times

model = YOLO(model_path, task='detect')
capture = cv2.VideoCapture(img_source)

distances, times = analyze_video(capture, model)
plt.plot(times, distances, marker='o', markersize=1, linestyle='-', color='b', linewidth=1.2)
plt.title('Distance between cat centroids (px) vs Time Graph')
plt.xlabel('Time')
plt.ylabel('Distance')
plt.show()