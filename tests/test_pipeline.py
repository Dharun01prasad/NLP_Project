import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from toxicity.data import binary_label, clean_frame, group_key, read_upload
from toxicity.model import augment_training, partition, predict, train_experiment
from toxicity.server import create_app


@pytest.fixture
def corpus():
    rows = []
    for label, phrase in [(0, 'nice helpful video thanks nanri'), (1, 'idiot insulting garbage abusive')]:
        for i in range(100):
            rows.append({'comment': f'{phrase} example {label} serial {i}', 'label': label})
    return clean_frame(pd.DataFrame(rows))[0]


def test_csv_quoted_text_and_labels():
    df, info = read_upload('comment,label\n"Hello, friend",0\n"abusive, insult",1\n"other",not-Tamil\n')
    assert list(df.comment) == ['Hello, friend', 'abusive, insult']
    assert info['excluded']['other_language'] == 1
    for bad in ['negative', 'positive', '10', 'unknown']:
        with pytest.raises(ValueError): binary_label(bad)
    assert binary_label('Offensive_Untargetede') == 1
    assert binary_label('Not_offensive') == 0
    assert binary_label('clean', {'clean': 0}) == 0


def test_duplicate_conflicts_and_test_preference():
    raw = pd.DataFrame([
        {'comment': 'Hello friend', 'label': 0, 'split': 'train'},
        {'comment': 'hello friend', 'label': 0, 'split': 'test'},
        {'comment': 'same bad', 'label': 0, 'split': 'train'},
        {'comment': 'same bad', 'label': 1, 'split': 'train'},
        {'comment': 'An insult', 'label': 1, 'split': 'train'},
    ])
    df, info = clean_frame(raw)
    assert len(df) == 2
    assert info['excluded']['conflicting_duplicate_rows'] == 2
    assert df.loc[df.label.eq(0), 'split'].iloc[0] == 'test'


def test_no_group_leakage_and_train_only_augmentation(corpus):
    parts, _ = partition(corpus)
    for key, a in parts.items():
        for other, b in parts.items():
            if key != other: assert not set(a.group) & set(b.group)
    augmented = augment_training(parts['train'])
    assert set(augmented.id).issubset(set(parts['train'].id))
    assert group_key('Heeellooo!') == group_key('Helo')
    assert group_key('தமிழ் நல்லது') == group_key('தமிழ் நல்லது!')


def test_training_scoring_and_api(corpus, tmp_path):
    info = {'name': 'synthetic test corpus', 'labels_verified': False, 'revision': 'test'}
    bundle = train_experiment(corpus, info, tmp_path/'models', tmp_path/'reports', augment=True)
    results = predict(bundle, ['nice helpful video thanks', 'idiot insulting garbage'])
    assert results[0]['probability'] < results[1]['probability']
    assert all(0 <= row['score'] <= 10 for row in results)
    assert results[1]['evidence']
    assert bundle['report']['human_validated'] is False
    assert len(bundle['report']['experiments']) == 3
    assert list((tmp_path/'reports').glob('run_*.json'))
    client = TestClient(create_app(tmp_path))
    assert client.get('/api/status').json()['model_ready']
    response = client.post('/api/analyze', json={'comments': ['nice helpful video thanks', 'idiot garbage']})
    assert response.status_code == 200
    assert response.json()['summary']['analyzed'] == 2
    assert response.json()['human_validated'] is False
    assert client.post('/api/analyze', json={'comments': [' ']}).status_code == 422
    assert client.post('/api/analyze', json={'comments': ['ok'] * 501}).status_code == 422
    assert client.post('/api/analyze', json={'comments': ['ok']}, headers={'origin': 'https://evil.example'}).status_code == 403
    assert client.post('/api/train', json={}, headers={'origin': 'chrome-extension://' + 'a'*32}).status_code == 403
    assert client.get('/api/export/unknown').status_code == 404
    assert client.get('/').status_code == 200


def test_missing_model_never_returns_fake_results(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert not client.get('/api/status').json()['model_ready']
    assert client.post('/api/analyze', json={'comments': ['anything']}).status_code == 503


def test_reviews_only_change_training_data_and_preserve_test(corpus, tmp_path):
    app = create_app(tmp_path)
    store = app.state.store
    store.install_data(corpus, {'name': 'test', 'class_counts': {}, 'excluded': {}})
    df = store.dataframe()
    before_test = set(df[df.split.eq('test')].id)
    row = df[df.split.eq('train')].iloc[0]
    info = store.read_json('data/info.json')
    store.write_json('reports/audit.json', {'rows': [{'id': row.id}], 'revision': info['revision']})
    client = TestClient(app)
    response = client.post('/api/reviews', json={'revision': info['revision'], 'decisions': [{'id': row.id, 'label': 1-int(row.label)}]})
    assert response.status_code == 200
    after = store.dataframe()
    assert set(after[after.split.eq('test')].id) == before_test
    assert int(after[after.id.eq(row.id)].label.iloc[0]) == 1-int(row.label)
    assert store.read_json('data/info.json')['corrected_count'] == 1
    assert client.post('/api/reviews', json={'revision': info['revision'], 'decisions': [{'id': row.id, 'label': 0}]}).status_code == 409


def test_audit_receives_training_rows_only(corpus, monkeypatch):
    import toxicity.audit as audit
    parts, _ = partition(corpus)
    seen = []
    def fake_find(*, labels, pred_probs, **kwargs):
        seen.append((len(labels), pred_probs.copy()))
        return np.array([0])
    monkeypatch.setattr(audit, 'find_label_issues', fake_find)
    result = audit.audit_labels(corpus)
    assert result['training_rows'] == len(parts['train'])
    assert result['rows'][0]['id'] in set(parts['train'].id)
    assert np.allclose(seen[0][1].sum(axis=1), 1)


def test_bad_import_preserves_previous_dataset(corpus, tmp_path):
    app = create_app(tmp_path)
    store = app.state.store
    info = store.install_data(corpus, {'name': 'valid', 'excluded': {}})
    client = TestClient(app)
    job = client.post('/api/datasets/import', json={'content': 'comment,label\nhello,negative\nbye,positive\n'}).json()
    for _ in range(100):
        response = client.get('/api/jobs/' + job['id']).json()
        if response['status'] != 'running': break
        time.sleep(.02)
    assert response['status'] == 'failed'
    assert store.read_json('data/info.json')['revision'] == info['revision']
