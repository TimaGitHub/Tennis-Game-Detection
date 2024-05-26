import cv2
import numpy as np


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

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray,100, 200, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 1, minLineLength=10, maxLineGap=1)

    intersection = find_intersection(lines)