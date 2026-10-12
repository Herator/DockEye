import csv
import json

import pytest

from dataset_tool import build_dataset, load_manifest, validate_dataset
from evaluate import evaluate, match_boxes
from tests.test_pipeline import write_video


def manifest(tmp_path):
    classes = ['boat','person','person_fishing']
    clips = []
    for split in ['train','val','test']:
        video = tmp_path/f'{split}.mp4'
        write_video(video)
        clips.append({'id':split,'group':split,'split':split,'video':str(video),
              'frames':[{'index':i+3,'objects':[{'label':label,'box':[10,10,20,30]}]}
                        for i,label in enumerate(classes)]})
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'clips':clips}))
    return path


def test_build_and_validate_dataset_with_group_provenance(tmp_path):
    path = manifest(tmp_path)
    yaml_path = build_dataset(path,tmp_path/'dataset')
    assert validate_dataset(yaml_path)['names'] == ['boat','person','person_fishing']
    assert len(list((tmp_path/'dataset/images').rglob('*.jpg'))) == 9


def test_group_leakage_rejected_before_training(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data['clips'][1]['group'] = data['clips'][0]['group']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='leakage'):
        load_manifest(path)


def test_missing_class_rejected(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data['clips'][0]['frames'].pop()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='every class'):
        build_dataset(path,tmp_path/'dataset')


def test_same_video_cannot_be_disguised_as_different_groups(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data['clips'][1]['video'] = data['clips'][0]['video']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='same video'):
        load_manifest(path)


def test_matching_requires_correct_class_and_one_to_one():
    expected = [{'label':'person','box':[0,0,20,20]}]
    predicted = expected*2+[{'label':'person_fishing','box':[0,0,20,20]}]
    assert match_boxes(expected,predicted) == (1,2,0)


def test_empty_hour_and_missing_frame_evaluation(tmp_path):
    root = tmp_path/'run'
    root.mkdir()
    video = tmp_path/'empty.mp4'
    path = tmp_path/'manifest.json'
    clip = {'id':'empty','group':'empty','split':'test','video':str(video),
            'empty_dock':True,'events':[], 'frames':[{'index':0,'objects':[]}]}
    path.write_text(json.dumps({'clips':[clip]}))
    (root/'summary.json').write_text(json.dumps({'mode':'model','frames_processed':36000,
        'fps':10,'boat_count':0,'parameters':{'video':str(video)}}))
    (root/'predictions.jsonl').write_text(json.dumps({'frame':0,'tracks':[]})+'\n')
    (root/'events.csv').write_text('source,event_type,start_seconds,end_seconds\n')
    report = evaluate(path,'empty',root)
    assert report['duration_hours'] == 1
    assert report['false_events_per_hour'] == 0
    assert report['classes']['person']['precision'] is None
    clip['frames'][0]['index'] = 1
    path.write_text(json.dumps({'clips':[clip]}))
    with pytest.raises(ValueError,match='not processed'):
        evaluate(path,'empty',root)
