import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Define and parse user input arguments

# parser = argparse.ArgumentParser()
# parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
#                     required=True)
# parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
#                     image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")',
#                     required=True)
# parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
#                     default=0.5)
# parser.add_argument('--resolution', help='Resolution in WxH to resize inference results to (example: "640x480"), \
#                     otherwise, match source resolution. Affects both the saved output and any displayed window.',
#                     default=None)
# parser.add_argument('--output', help='Where to save annotated results: a directory for image/folder sources \
#                     (one file per input image, same filename), or a video file path for video/camera sources \
#                     (".avi"/".mp4"; extension picked for you if omitted). Results are always saved, with or \
#                     without a display available.',
#                     default='output')
# parser.add_argument('--record', help='(Deprecated) Results are now always saved automatically to --output, \
#                     so this flag no longer does anything. Kept for backward compatibility.',
#                     action='store_true')
#
# args = parser.parse_args()


# Parse user inputs
model_path = 'best.pt' # args.model
# img_source = './video.mov' #args.source
img_source = '/Users/vyankov/Documents/Datasets/cat-fights/TMP/IMG_8116.mov'

min_thresh = 0.5 # float(args.thresh)
user_res = None # args.resolution
output = './output'

# if args.record:
#     print('NOTE: --record is deprecated and has no effect — results are now always saved automatically to --output.')

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Use Apple's Metal (MPS) backend when available so inference runs on the GPU instead of CPU;
# fall back to CPU on machines without it (e.g. Intel Macs, or a torch build without MPS).
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Running inference on device: {device}')

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Detect whether a GUI window can actually be shown. Headless environments (e.g. Colab,
# SSH sessions, CI runners) have no X display on Linux, so cv2.imshow()/waitKey() would
# crash the whole process instead of raising a catchable error. Skip them proactively.
headless = ('DISPLAY' not in os.environ) and sys.platform.startswith('linux')
if headless:
    print('No display detected — running headless. Results will be saved to disk instead of shown in a window.')

# Set up where annotated results get saved. This always happens, regardless of whether
# a display is available, so results from a headless run (e.g. Colab) are still usable.
save_video = source_type in ('video', 'usb', 'picamera')
if save_video:
    video_out_path = output if output.lower().endswith(('.avi', '.mp4')) else output + '.mp4'
    out_dir = os.path.dirname(video_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    video_writer = None  # created lazily once the first frame's size is known
else:
    img_out_dir = output
    os.makedirs(img_out_dir, exist_ok=True)

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

# elif source_type == 'picamera':
#     from picamera2 import Picamera2
#     cap = Picamera2()
#     cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
#     cap.start()

class_settings = [('cat_white', (255, 255, 255), 'Zeus'), ('cat_grey', (163, 163, 163), 'Dio')]
# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1

    elif source_type == 'video': # If source is a video, load next frame from video file
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break

    elif source_type == 'usb': # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera': # If source is a Picamera, grab frames using picamera interface
        frame = cap.capture_array()
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, conf=min_thresh, device=device, verbose=True) # default (conf=0.25, iou=0.7)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    white_found = False
    gray_found = False

    # Process detections in descending-confidence order rather than raw model output order
    order = np.argsort(-detections.conf.cpu().numpy())

    # Go through each detection and get bbox coords, confidence, and class
    for i in order:

        conf = detections[i].conf.item()
        if conf <= min_thresh:
            continue

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Workaround for duplicate boxes on the same cat (e.g. one on the face, one on the
        # whole body): there are only ever two cats, one white and one gray, so keep at most
        # one detection per class — now guaranteed to be the most confident one.
        if classname == 'cat_grey' and gray_found:
            continue
        if classname == 'cat_white' and white_found:
            continue

        if classname == 'cat_grey':
            gray_found = True
        elif classname == 'cat_white':
            white_found = True

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        found_cats = [cat for cat in class_settings if cat[0] == classname]
        color = found_cats[0][1]
        name = found_cats[0][2]
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

        label = f'{name}: {int(conf*100)}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4) # Draw label text

        # Basic example: count the number of objects in the image
        object_count = object_count + 1

    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4) # Draw framerate

    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,105), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 4) # Draw total number of detected objects

    # Save the annotated frame to disk (always, independent of whether a window is shown)
    if save_video:
        if video_writer is None:
            h, w = frame.shape[:2]
            video_fps = cap.get(cv2.CAP_PROP_FPS) if source_type == 'video' else 30
            if not video_fps or video_fps <= 0:
                video_fps = 30
            fourcc = cv2.VideoWriter_fourcc(*('mp4v' if video_out_path.lower().endswith('.mp4') else 'MJPG'))
            video_writer = cv2.VideoWriter(video_out_path, fourcc, video_fps, (w, h))
        video_writer.write(frame)
    else:
        cv2.imwrite(os.path.join(img_out_dir, os.path.basename(img_filename)), frame)

    # Show a window with results, if a display is actually available
    if not headless:
        cv2.imshow('YOLO detection results', frame) # Display image

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    # Skip entirely when headless — there's no window to receive keypresses, and waitKey() would crash without a display.
    if headless:
        key = -1
    elif source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if save_video:
    if video_writer is not None:
        video_writer.release()
        print(f'Saved annotated video to {video_out_path}')
else:
    print(f'Saved annotated images to {img_out_dir}/')
if not headless:
    cv2.destroyAllWindows()