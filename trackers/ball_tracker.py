import cv2
from ultralytics import YOLO
from tqdm.auto import tqdm
from trackers.model import BallTrackerNet
import torch
import torch.nn as nn
import numpy as np
from itertools import groupby
from scipy.spatial import distance
import pickle
import pandas as pd


class BallTrackerYolo:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y`', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        return [{1: x} for x in df_ball_positions.to_numpy().tolist()]

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def detect_frames(self, frames, read_from_stub = False, stub_path = None):

        ball_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in tqdm(frames, position=0):
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        return ball_detections


    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # draw bboxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            output_video_frames.append(frame)

        return output_video_frames


class BallTrackerTrackNet:
    def __init__(self, model_path):
        self.model = BallTrackerNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)

    def infer_model(self, frames):
        height = 360
        width = 640
        dists = [-1] * 2
        ball_track = [(None,None)]*2

        for num in tqdm(range(2, len(frames))):
            img = cv2.resize(frames[num], (width, height))
            img_prev = cv2.resize(frames[num-1], (width, height))
            img_preprev = cv2.resize(frames[num-2], (width, height))
            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32)/255.0
            #
            imgs = np.rollaxis(imgs, 2, 0)
            #
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = self.postprocess(output, scale=1)
            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] and ball_track[-2][0]:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)
        return ball_track, dists

    def remove_outliers(self, ball_track, dists, max_dist=100):
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
                ball_track[i] = (None, None)
                outliers.remove(i)
            elif dists[i - 1] == -1:
                ball_track[i - 1] = (None, None)
        return ball_track

    def split_track(self, ball_track, max_gap=4, max_dist_gap=80, min_track=5):

        list_det = [0 if x[0] else 1 for x in ball_track]
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]
        cursor = 0
        min_value = 0
        result = []
        for i, (k, l) in enumerate(groups):
            if (k == 1) & (i > 0) & (i < len(groups) - 1):
                dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + l])
                if (l >= max_gap) | (dist / l > max_dist_gap):
                    if cursor - min_value > min_track:
                        result.append([min_value, cursor])
                        min_value = cursor + l - 1
            cursor += l
        if len(list_det) - min_value > min_track:
            result.append([min_value, len(list_det)])
        return result

    def interpolation(self, coords):
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
        y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

        nons, yy = nan_helper(x)
        x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
        nans, xx = nan_helper(y)
        y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

        track = [*zip(x, y)]
        return track

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        ball_track, dists = self.infer_model(frames)
        ball_track = self.remove_outliers(ball_track, dists)
        subtracks = self.split_track(ball_track)
        for r in subtracks:
            ball_subtrack = ball_track[r[0]:r[1]]
            ball_subtrack = self.interpolation(ball_subtrack)
            ball_track[r[0]:r[1]] = ball_subtrack

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_track, f)
        return ball_track

    def postprocess(self, feature_map, scale=1):
        feature_map *= 255
        feature_map = feature_map.reshape((360, 640))
        feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
        x, y = None, None
        if circles is not None:
            if len(circles) == 1:
                x = circles[0][0][0] * scale
                y = circles[0][0][1] * scale
        return x, y

    def draw_bboxes(self, frames, ball_track):
        output_video_frames = []
        height, width = frames[0].shape[:2]
        trace = 7
        color_grid = [(26, 26, 235),
                      (116, 26, 235),
                      (228, 26, 235),
                      (235, 26, 172),
                      (235, 26, 89),
                      (238, 172, 26),
                      (123, 235, 26),
                      ]
        for num in range(len(frames)):
            frame = frames[num]
            for i in range(trace):
                if (num - i > 0):
                    if ball_track[num - i][0]:
                        x = int(ball_track[num - i][0] * width / 640)
                        y = int(ball_track[num - i][1] * height / 360)
                        frame = cv2.circle(frame, (x, y), radius=0, color=color_grid[i], thickness=10 - i)
                    else:
                        break
            output_video_frames.append(frame)
        return output_video_frames




