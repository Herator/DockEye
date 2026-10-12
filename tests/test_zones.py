import json

import numpy as np
import pytest

from zones import make_zones


def test_explicit_zone_scaling_preserves_source_file(tmp_path):
    path = tmp_path/'zones.json'
    zones = {'person':[[0,0],[199,0],[199,99]],'boat':[[0,0],[199,99],[0,99]]}
    original = json.dumps(zones)
    path.write_text(original)
    image = np.zeros((50,100,3),np.uint8)
    with pytest.raises(ValueError,match='do not fit'):
        make_zones(image,{},path,headless=True)
    scaled = make_zones(image,{},path,headless=True,reference_size=(200,100))
    assert scaled['person'] == [[0,0],[99,0],[99,49]]
    assert path.read_text() == original
