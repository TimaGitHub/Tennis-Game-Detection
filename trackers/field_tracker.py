from ultralytics import YOLO
import torch
import torch.nn as nn
import pickle
from tqdm.auto import tqdm

class FieldTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)

    def get_frames_with_field(self, frames, read_from_stub, stub_path):

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                out_frames = pickle.load(f)
            return out_frames

        out_frames = []

        for frame in tqdm(frames):
            if isinstance(self.model, nn.DataParallel):
                result = self.model.module.predict(frame, verbose=False)[0].boxes.cls.tolist()
            else:
                result = self.model.predict(frame, verbose=False)[0].boxesc.cls.tolist()

            if result == []:
                out_frames.append(0)
            else:
                out_frames.append(1)
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(out_frames, f)

        return out_frames