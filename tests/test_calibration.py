import numpy as np
import pytest

from calibration import Calibration
from tracker import ObjectTrack


def test_known_ground_coordinates_and_outside_plane():
    calibration = Calibration({'image_points':[[0,0],[100,0],[100,100],[0,100]],
                               'world_points_m':[[0,0],[10,0],[10,10],[0,10]],
                               'frame_size':[200,200]})
    assert calibration.to_metres((50,50)) == pytest.approx([5,5])
    assert calibration.to_metres((150,150)) is None
    track = ObjectTrack(1,(40,10,20,40),'person',0.9,0,0)
    assert calibration.ground_position(track,(200,200,3)) == pytest.approx([5,5])
    with pytest.raises(ValueError,match='frame_size'):
        calibration.ground_position(track,(100,100,3))


def test_collinear_calibration_rejected():
    with pytest.raises(ValueError,match='degenerate'):
        Calibration({'image_points':[[0,0],[1,1],[2,2],[3,3]],
                     'world_points_m':[[0,0],[1,1],[2,2],[3,3]],'frame_size':[100,100]})
