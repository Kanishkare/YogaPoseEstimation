from __future__ import division  # MUST be at the top

import sys
import os
# Make the `helpers` directory importable when this script is run from the project root.
# `helpers` lives under the `yoga` folder (./yoga/helpers). Adding that folder to
# sys.path allows `from helpers.visualize_yoga import ...` to succeed.
sys.path.append(os.path.join(os.path.dirname(__file__), 'yoga'))

import argparse
import time
import cv2
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
import matplotlib.pyplot as plt
import gluoncv as gcv
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
# from helpers.visualize_yoga import cv_plot_keypoints  # removed (TensorFlow dependency)

# Minimal TensorFlow-free replacement for helpers.visualize_yoga.cv_plot_keypoints
def cv_plot_keypoints(frame_rgb, pred_coords, confidence, class_IDs, _unused, scores,
                      box_thresh=0.5, keypoint_thresh=0.15):
	"""
	Simple replacement used by this script.
	- frame_rgb: RGB numpy image (H,W,3) or MXNet NDArray
	- pred_coords: (N, K, 2) array or MXNet NDArray of keypoint coordinates (x,y)
	- confidence: (N, K) array/NDArray of keypoint confidences
	- class_IDs, _unused, scores: passed through from detector (scores can be NDArray)
	Returns: (annotated_rgb_image, pose_summary_string_or_None)
	"""
	import numpy as _np
	import cv2 as _cv2

	# Convert MXNet NDArrays to numpy if needed
	if hasattr(pred_coords, 'asnumpy'):
		coords = pred_coords.asnumpy()
	else:
		coords = _np.array(pred_coords) if pred_coords is not None else _np.zeros((0,0,2))

	if hasattr(confidence, 'asnumpy'):
		conf = confidence.asnumpy()
	else:
		# shape fallback: (N, K)
		conf = _np.array(confidence) if confidence is not None else _np.zeros((coords.shape[0], coords.shape[1] if coords.size else 0))

	if hasattr(scores, 'asnumpy'):
		scores_np = scores.asnumpy()
	else:
		scores_np = _np.array(scores) if scores is not None else _np.zeros((coords.shape[0],))

	# Ensure an RGB copy we can draw on
	out = frame_rgb.copy() if frame_rgb is not None else None
	if out is None or coords.size == 0:
		return frame_rgb, None

	pose_texts = []
	for i in range(coords.shape[0]):
		# handle scores shape (N,1) or (N,)
		try:
			sc = float(scores_np[i].max()) if scores_np.ndim > 1 else float(scores_np[i])
		except Exception:
			sc = 0.0
		if sc < box_thresh:
			continue
		detected_kps = 0
		for j in range(coords.shape[1]):
			x, y = coords[i, j]
			conf_j = float(conf[i, j]) if conf.size else 0.0
			if conf_j >= keypoint_thresh:
				_cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
				detected_kps += 1
		pose_texts.append(f'kp:{detected_kps}')

	pose_summary = pose_texts[0] if len(pose_texts) > 0 else None
	return out, pose_summary

import pandas as pd
# Use cv2.VideoCapture instead of imutils.VideoStream to avoid extra dependency

# ----------- Setup MXNet context and models -----------
ctx = mx.cpu()

# Detector for human
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)
detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
detector.hybridize()

# Pose estimator
estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
estimator.hybridize()

# ----------- Parse command line arguments -----------
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--vid', type=str, required=False, help='video path')
ap.add_argument('-o', '--outfile', default='output.avi', help='outfile path')
args = vars(ap.parse_args())

using_vid_file = False

if args['vid'] is not None:  # Using video file
    using_vid_file = True
    print('[INFO] Using video file...')
    vid_path = os.path.join(os.getcwd(), args['vid'])
    vs = cv2.VideoCapture(vid_path)
    vid_writer = cv2.VideoWriter(args['outfile'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (500, 280), True)
else:  # Using webcam (use cv2.VideoCapture to avoid imutils dependency)
    print('[INFO] Using webcam...')
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)

# ----------- Main processing function -----------
def main():
    start = time.time()
    count = 0
    pose = None
    skip_frame = False

    while True:
        # cv2.VideoCapture.read() returns (ret, frame) for both file and webcam
        exists, frame = vs.read()
        if not exists or frame is None:
            break

        # Mirror image
        frame = np.fliplr(frame)
        count += 1
        # Resize to width=280 preserving aspect ratio
        h, w = frame.shape[:2]
        target_w = 280
        scale = target_w / float(w)
        new_h = max(1, int(h * scale))
        frame = cv2.resize(frame, (target_w, new_h))
        frame_nd = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

        if not skip_frame:
            x, frame_transformed = gcv.data.transforms.presets.ssd.transform_test(frame_nd, short=512, max_size=280)
            x = x.as_in_context(ctx)

            class_IDs, scores, bounding_boxs = detector(x)
            pose_input, upscale_bbox = detector_to_simple_pose(frame_transformed, class_IDs, scores, bounding_boxs,
                                                               output_shape=(128, 96), ctx=ctx)
            if len(upscale_bbox) > 0:
                predicted_heatmap = estimator(pose_input)
                pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
                img, pose = cv_plot_keypoints(frame_transformed, pred_coords, confidence, class_IDs, None, scores,
                                              box_thresh=0.5, keypoint_thresh=0.15)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                # ensure img exists even when no bbox is found
                img = frame.copy()
            skip_frame = True
        else:
            skip_frame = False

        # Force output size to (500, 280)
        img = cv2.resize(img, (500, 280))
        if pose:
            cv2.putText(img, '{}'.format(pose), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'No Pose Detected', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if using_vid_file:
            vid_writer.write(img)

        cv2.imshow('Yoga Pose Estimation', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    # Release resources
    if vs is not None:
        vs.release()
    if using_vid_file:
        vid_writer.release()
    stop = time.time()
    print("FPS: {:.2f}".format(count / (stop - start)))


if __name__ == '__main__':
    main()
