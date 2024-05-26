import cv2
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    # get_foot_position,
    # get_closest_keypoint_index,
    # get_height_of_bbox,
    # measure_xy_distance,
    # get_center_of_bbox,
    # measure_distance
)
import numpy as np
import pickle
import pandas as pd


class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 550
        self.buffer = 100
        self.padding_court_x = 20
        self.padding_court_y = 50

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()


    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court_x
        self.court_start_y = self.start_y + self.padding_court_y
        self.court_end_x = self.end_x - self.padding_court_x
        self.court_end_y = self.end_y - self.padding_court_y
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                                )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 28

        # point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        # #point 8
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        # #point 10
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_court(self, frame):
        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        return frame

    def draw_mini_court(self, frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(frames, player_detections):
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            self.draw_player_point(frame, player_dict)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def draw_player_point(self, frame, player_dict):
        x_1 = self.drawing_key_points[4] + int(abs(self.drawing_key_points[4] - self.drawing_key_points[6]) * player_dict[0][0])
        y_1 = self.drawing_key_points[5] + player_dict[0][1] * abs(self.drawing_key_points[1] - self.drawing_key_points[5])

        x_2 = self.drawing_key_points[0] + int(abs(self.drawing_key_points[0] - self.drawing_key_points[2]) * player_dict[1][0])
        y_2 = self.drawing_key_points[1] + player_dict[1][1] * abs(self.drawing_key_points[1] - self.drawing_key_points[5])
        cv2.circle(frame, (int(x_1), int(y_1)), 5, (128, 0, 0) , -1)
        cv2.circle(frame, (int(x_2), int(y_2)), 5, (128, 0, 0) , -1)


    def point_players_on_mini_court(self, shape, player_detections, court_keypoints):
        points_x = court_keypoints[0::2]
        points_y = court_keypoints[1::2]
        points = list(zip(points_x, points_y))
        points = sorted(points, key = lambda x: x[0])
        left_down_point = points[0]
        right_down_point = points[-1]
        points = [x for x in points if x[1] < shape[0] * 0.6]
        points = sorted(points, key=lambda x: x[0])
        left_top_point = points[0]
        right_top_point = points[-1]

        left_down_point = tuple(map(int, left_down_point))
        right_down_point = tuple(map(int, right_down_point))
        left_top_point = tuple(map(int, left_top_point))
        right_top_point = tuple(map(int, right_top_point))

        dist = {}
        keys = list(player_detections[10].keys())
        for key, item in player_detections[0].items():
            dist[key] = (item[0] - left_down_point[0]) ** 2 + (item[1] - left_down_point[1]) ** 2
        if dist[keys[0]] < dist[keys[1]]:
            first_player, second_player = keys[0], keys[1]
        else:
            first_player, second_player = keys[1], keys[0]
        # first player - down player

        output_player_detections = []
        for detection in player_detections:
            temp = {}
            for key, item in detection.items():
                temp[key] = [int((item[0] + item[2]) / 2), item[3]]
            abs_distance_1 = abs(left_down_point[0] - right_down_point[0])
            distance_1 = (temp[first_player][0] - left_down_point[0]) / abs_distance_1
            abs_distance_2 = abs(left_top_point[0] - right_top_point[0])
            distance_2 = (temp[second_player][0] - left_top_point[0]) / abs_distance_2

            abs_distance = abs(left_down_point[1] - left_top_point[1])
            temp[second_player][1] = (temp[second_player][1] - left_top_point[1]) / abs_distance
            temp[first_player][1] = (temp[first_player][1] - left_down_point[1]) / abs_distance
            if temp[first_player][1] > 0:
                temp[first_player][1] *= 0.5

            output_player_detections.append([(distance_1, temp[first_player][1]), (distance_2, temp[second_player][1])])

        return output_player_detections


    def track_hits_bounces(self, frames, path_detections = "./tracker_stubs/_tracknet_ball_detections.pkl", path_model = "./models/hits_bounces_detector.pkl"):
        with open(path_detections, 'rb') as f:
            ball_positions = pickle.load(f)
        ball_positions = np.array(ball_positions, dtype='float')
        ball_positions = pd.DataFrame(ball_positions).interpolate().to_numpy()
        ball_positions[:, 0], ball_positions[:, 1] = np.int_(ball_positions[:, 0] * 1920 / 640), np.int_(ball_positions[:, 1] * 1080 / 360)
        ball_positions = pd.DataFrame(ball_positions, columns=('x', 'y'))
        with open(path_model, 'rb') as f:
            model = pickle.load(f)

        def re(df):
            df = df.copy()
            df['vx'] = df['x'].diff()[1:]
            df['vy'] = df['y'].diff()[1:]
            df['v'] = np.sqrt(df['vx'] ** 2 + df['vy'] ** 2)
            df['ax'] = df['vx'].diff()[1:]
            df['ay'] = df['vy'].diff()[1:]
            df['x_prev'] = df['x'].shift(1)
            df['y_prev'] = df['y'].shift(1)
            df['vx_prev'] = df['vx'].shift(1)
            df['vy_prev'] = df['vy'].shift(1)
            # Положение, скорость и ускорение на следующем шаге времени
            df['x_next'] = df['x'].shift(-1)
            df['y_next'] = df['y'].shift(-1)
            df['vx_next'] = df['vx'].shift(-1)
            df['vy_next'] = df['vy'].shift(-1)
            df['x_next_2'] = df['x'].shift(-2)
            df['y_next_2'] = df['y'].shift(-2)
            df['vx_next_2'] = df['vx'].shift(-2)
            df['vy_next_2'] = df['vy'].shift(-2)
            df = df.dropna()
            return df

        ball_positions = re(ball_positions)
        pred = model.predict(ball_positions)
        pred = list(zip(np.where(pred != 0)[0].tolist(), pred[pred != 0].tolist()))

        new_predictions = []
        while pred:
            item = list(pred.pop(0))
            while pred != [] and pred[0][0] - item[0] < 10 and item[1] == pred[0][1]:
                item[0] = int((item[0] + pred.pop(0)[0]) / 2)
            new_predictions.append(item)

        for frame_index, pattern in new_predictions:
            if pattern == 1:
                for i in range(10):
                    cv2.putText(frames[frame_index + i], "HIT", (300, 350), cv2.FONT_HERSHEY_TRIPLEX, 2, (128, 0, 0), 4)
            else:
                for i in range(10):
                    cv2.putText(frames[frame_index + i], "BOUNCE", (140, 700), cv2.FONT_HERSHEY_TRIPLEX, 2, (128, 0, 0), 4)

        return frames




