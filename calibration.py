"""Optional dock-plane coordinates; a homography cannot measure body height."""
import json
from pathlib import Path

import cv2
import numpy as np


class Calibration:
    def __init__(self, configuration):
        self.image_points = np.asarray(configuration['image_points'],np.float32)
        self.world_points = np.asarray(configuration['world_points_m'],np.float32)
        if (self.image_points.shape != self.world_points.shape or self.image_points.ndim != 2
                or self.image_points.shape[1] != 2 or len(self.image_points) < 4
                or not np.isfinite(self.image_points).all() or not np.isfinite(self.world_points).all()):
            raise ValueError('Calibration needs four or more finite matching 2D points')
        self.matrix, _ = cv2.findHomography(self.image_points,self.world_points,method=0)
        if self.matrix is None or not np.isfinite(self.matrix).all() or abs(np.linalg.det(self.matrix)) < 1e-12:
            raise ValueError('Calibration points are degenerate')
        self.frame_size = tuple(configuration['frame_size'])
        if len(self.frame_size) != 2 or min(self.frame_size) <= 0:
            raise ValueError('Calibration needs a positive frame_size [width,height]')

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text()))

    def to_metres(self, point):
        # Only interpolate inside measured points on the same physical plane.
        hull = cv2.convexHull(self.image_points)
        if cv2.pointPolygonTest(hull,tuple(map(float,point)),False) < 0:
            return None
        result = cv2.perspectiveTransform(np.asarray([[point]],np.float32),self.matrix)[0,0]
        return [float(value) for value in result]

    def ground_position(self,track,frame_shape):
        if track.label not in ('person','person_fishing','person_fishing_candidate'):
            return None
        height,width = frame_shape[:2]
        if (width,height) != self.frame_size:
            raise ValueError('Calibration frame_size differs from video; recalibrate explicitly')
        x,y,w,h = track.box
        return self.to_metres((x+w/2,y+h))
