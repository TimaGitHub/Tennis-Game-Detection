from utils import (read_video,
                   save_video,
                   split_frames_by_ones)
from trackers import PlayerTracker, BallTrackerYolo, BallTrackerTrackNet, FieldTracker
from court_line_detector import CourtLineDetector
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import cv2
from mini_court import MiniCourt
import time

def main():
    # read video
    input_video_path = "input_videos/input_video.mp4"
    print("Reading the video...")
    start = time.time()
    video_frames = read_video(input_video_path)
    print("Reading is done ...", round(time.time() - start, 1))
    print('')
    player_tracker = PlayerTracker(model_path='yolov8x.pt', model_racket_path="models/racket_last_v8.pt")
    ball_tracker_1 = BallTrackerYolo(model_path="models/yolo5_best.pt")
    ball_tracker_2 = BallTrackerTrackNet(model_path="models/tracknet_model_best.pt", )
    court_line_detector = CourtLineDetector(model_path="models/keypoints_model.pth")
    field_tracker = FieldTracker(model_path="models/best court detector.pt")
    mini_court = MiniCourt(video_frames[0])


    # detecting field in frames
    print("Detecting field in frames...")
    start = time.time()
    frames_with_fields = field_tracker.get_frames_with_field(video_frames,
                                                            read_from_stub=True,
                                                            stub_path="tracker_stubs/frames_with_field.pkl")

    for i in range(2, len(frames_with_fields) -3):
        if frames_with_fields[i] != frames_with_fields[i-1] and frames_with_fields[i] != frames_with_fields[i-2]\
            and frames_with_fields[i] != frames_with_fields[i + 1] and frames_with_fields[i] != frames_with_fields[i+2]:
            frames_with_fields[i] = frames_with_fields[i-1]
    print('')
    print("Detecting is done ...", round(time.time() - start, 1))
    print('')
    final_frames = []
    list_of_frames = split_frames_by_ones(frames_with_fields, video_frames)
    for (pattern, frames) in list_of_frames:
        if pattern == 1:

            # detecting keypoints

            print("Detecting keypoints...")
            start = time.time()
            court_keypoints = court_line_detector.predict(frames[20])

            print("Keypoints are detected ", round(time.time() - start, 1))
            print('')

            # detecting players
            print("Detecting player...")
            start = time.time()
            player_detections = player_tracker.detect_frames(frames, court_keypoints,
                                                            read_from_stub=True,
                                                            stub_path="tracker_stubs/new_player_detections.pkl")
            print("Players are detected ", round(time.time() - start, 1))
            print('')
            # detecting ball
            # ball_detections = ball_tracker_1.detect_frames(frames,
            #                                                 read_from_stub=True,
            #                                                 stub_path="tracker_stubs/ball_detections.pkl")
            # ball_detections = ball_tracker_1.interpolate_ball_positions(ball_detections)
            print("Detecting ball...")
            start = time.time()
            ball_detections = ball_tracker_2.detect_frames(frames,
                                                            read_from_stub=True,
                                                            stub_path="tracker_stubs/_tracknet_ball_detections.pkl")
            print("Ball is detected ", round(time.time() - start, 1))
            print('')

            # draw

            # draw court lines
            print("Drawing court lines...")
            start = time.time()
            output_video_frames = court_line_detector.draw_keypoints_on_video(frames, court_keypoints)
            print("Court lines are done ", round(time.time() - start, 1))
            print('')


            # draw ball bounding boxes

            print("Drawing ball...")
            start = time.time()
            # output_video_frames = ball_tracker_1.draw_bboxes(output_video_frames, ball_detections)
            output_video_frames = ball_tracker_2.draw_bboxes(output_video_frames, ball_detections)
            print("Ball is done ", round(time.time() - start, 1))
            print('')

            # draw player bounding boxes
            print("Drawing player bounding boxes...")
            start = time.time()
            output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections)
            print("player bounding boxes are done ", round(time.time() - start, 1))
            print('')

            # get player positions on mini court
            print("Getting player positions on mini court...")
            start = time.time()
            output_player_detections = mini_court.point_players_on_mini_court(output_video_frames[0].shape, player_detections, court_keypoints)
            print("Getting is done ", round(time.time() - start, 1))
            print('')

            # draw frame rate on top left corner
            for i, frame in enumerate(output_video_frames):
                cv2.putText(frame, f"Frame: {i // 2}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            print("Tracking hits and bounces...")
            start = time.time()
            output_video_frames = mini_court.track_hits_bounces(output_video_frames)
            print("Tracking is done ", round(time.time() - start, 1))
            print('')

            # draw mini court
            print("Drawing mini court...")
            start = time.time()
            output_video_frames = mini_court.draw_mini_court(output_video_frames, output_player_detections)
            print("Mini court is done ", round(time.time() - start, 1))
            print('')
            final_frames += output_video_frames
        else:
            final_frames += frames

    # save_videos
    print("Saving video...")
    start = time.time()
    save_video(final_frames, "output_videos/output_video.avi")
    print("Saving is done ", round(time.time() - start, 1))
    print('')


if __name__ == "__main__":
    main()