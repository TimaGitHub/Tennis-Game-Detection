import torch
import torch.nn as nn
from torchvision.transforms import transforms
import torchvision.models as models
import cv2
import numpy as np


class CourtLineDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    def predict(self, image):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])
            min_size = 35
            while min_size < 100:
                try:
                    new_x, new_y = get_coords(img_rgb[max(0, y - min_size): y + min_size, max(0, x - min_size): x + min_size])
                    keypoints[i], keypoints[i + 1] = max(0, x - min_size) + new_x, max(0, y - min_size) + new_y
                    break
                except:
                    min_size += 3

        return keypoints


    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 0), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            output_video_frames.append(self.draw_keypoints(frame, keypoints))

        return output_video_frames

def find_intersection(lines):
    if len(lines) < 2:
        return None
    A = []
    B = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            A.append([y2 - y1, x1 - x2])
            B.append([(y2 - y1) * x1 + (x1 - x2) * y1])
    A = np.array(A)
    B = np.array(B)
    intersection = np.linalg.lstsq(A, B, rcond=None)[0]
    return int(intersection[0]), int(intersection[1])

def get_coords(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    edges = cv2.Canny(gray,100, 200, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1, minLineLength=8, maxLineGap=8)

    return find_intersection(lines)
