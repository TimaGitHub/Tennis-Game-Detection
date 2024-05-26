import cv2
import gc

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        del frame
        gc.collect()
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    #print(f'Video saved to {output_video_frames}')

def split_frames_by_ones(primary_list, frames):

    result = []
    current_value = primary_list[0]
    current_sublist = [frames[0]]

    for i in range(1, len(primary_list)):
        if primary_list[i] == current_value:
            current_sublist.append(frames[i])
        else:
            result.append((current_value, current_sublist))
            current_value = primary_list[i]
            current_sublist = [frames[i]]

    result.append((current_value, current_sublist))
    return result