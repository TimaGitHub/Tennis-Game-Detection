import cv2
from ultralytics import YOLO
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import pickle
import warnings
from utils import measure_distance, get_center_of_bbox


class PlayerTracker:
    def __init__(self, model_path, model_racket_path):
        self.model = YOLO(model_path)
        self.model_racket = YOLO(model_racket_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)
        self.model_racket = self.model_racket.to(self.device)
        self.model_racket = nn.DataParallel(self.model_racket)
        self.ids = []

    def detect_frame(self, frame, index):
        if isinstance(self.model, nn.DataParallel):
            results = self.model.module.track(frame, persist = True, verbose=False)[0]
        else:
            results = self.model.track(frame, persist = True, verbose=False)[0]
        #results = self.model.track(frame, persist = True, verbose=False)[0]
        id_name_dict = results.names
        player_dict = {}

        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == 'person':
                player_dict[track_id] = result
                if index < 20 and track_id not in self.ids:
                    x, y, w, h = box.xywh.tolist()[0]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    x, y, w, h = int(max(0, x - 1.5 * w)), int(min(frame.shape[1] - 1, x + 1.5 * w)), int(max(0, y - 1.5 * h)), int(min(frame.shape[0] - 1, y + 1.5 * h))
                    if isinstance(self.model_racket, nn.DataParallel):
                        temp_result = self.model_racket.module.predict(frame[w: h, x: y], verbose=False, conf = 0.4)[0].boxes.cls
                    else:
                        temp_result = self.model_racket.predict(frame[w: h, x: y], verbose=False, conf = 0.4)[0].boxes.cls
                    if temp_result is not None:
                        temp_result = temp_result.tolist()
                        if len(temp_result) == 1:
                            temp_result = int(temp_result[0])
                            if temp_result == 4:
                                self.ids.append(track_id)

        return player_dict

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[10]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # sort the distances in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, court_keypoints, read_from_stub = False, stub_path = None):

        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for index, frame in enumerate(tqdm(frames, position=0)):
            player_dict = self.detect_frame(frame, index)
            player_detections.append(player_dict)

        players = []
        for id in set(self.ids):
            x_shifts = [x[id][0] for x in player_detections[5:20]]
            y_shifts = [x[id][1] for x in player_detections[5:20]]
            if max(x_shifts) - min(x_shifts) > 5 and max(y_shifts) - min(y_shifts) > 5:
                players.append(id)

        if len(players) != 2:
            warnings.warn(f"Something went wrong with players detection, detected {len(players)} players", RuntimeWarning)
            player_detections = self.choose_and_filter_players(court_keypoints, player_detections)
        else:
            for i in range(len(player_detections)):
                player_detections[i] = {players[0]: player_detections[i][players[0]],
                                        players[1]: player_detections[i][players[1]]}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # draw bboxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)

        return output_video_frames


