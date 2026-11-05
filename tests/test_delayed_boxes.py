from src.events import boxes_at


def res(t, box, tid=1):
    return (t, {"person": {tid: box}, "boat": {}})


def test_midway_gives_midpoint():
    out = boxes_at([res(1.0, (0, 0, 10, 10)), res(1.2, (20, 40, 30, 50))], 1.1)
    assert out["person"][1] == (10, 20, 20, 30)


def test_only_before_is_used_as_is():
    out = boxes_at([res(1.0, (5, 6, 7, 8))], 1.3)
    assert out["person"][1] == (5, 6, 7, 8)


def test_track_missing_in_next_result_is_not_moved():
    out = boxes_at([res(1.0, (5, 6, 7, 8)), res(1.2, (50, 60, 70, 80), tid=2)], 1.1)
    assert out["person"] == {1: (5, 6, 7, 8)}


def test_stale_results_are_dropped():
    assert boxes_at([res(1.0, (5, 6, 7, 8))], 2.5) == {}
    assert boxes_at([], 1.0) == {}
    assert boxes_at([res(2.0, (5, 6, 7, 8))], 1.0) == {}  # nothing before t

