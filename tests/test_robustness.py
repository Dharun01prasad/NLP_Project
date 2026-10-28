import json

import numpy as np
import pandas as pd
import pytest

from toxicity.data import clean_frame, group_key, script_bucket, word_tokens
from toxicity.evaluation import evaluate_cases, sliced_metrics
from toxicity.inference import load_bundle
from toxicity.model import build_pipeline, metrics, pick_threshold, predict
from toxicity.train_transformer import fit_temperature


def test_tamil_words_keep_vowel_signs_and_pulli():
    text = 'வணக்கம் நன்றி! தமிழ் மொழி'
    assert word_tokens(text) == ['வணக்கம்', 'நன்றி', 'தமிழ்', 'மொழி']
    analyzer = build_pipeline('word').named_steps['features'].build_analyzer()
    assert 'வணக்கம் நன்றி' in analyzer(text)
    assert group_key('கல்') != group_key('கால்')
    assert group_key('தமிழ்!') == group_key('தமிழ்')
    assert script_bucket('vanakkam') == script_bucket('hello') == 'latin_only'


def test_import_preserves_source_and_language_and_excludes_nan():
    frame, info = clean_frame(pd.DataFrame([
        {'comment': 'hello', 'label': 0, 'language': 'english', 'source': 'reviewed'},
        {'comment': 'an insult', 'label': 1, 'language': 'english', 'source': 'reviewed'},
        {'comment': np.nan, 'label': 0},
    ]))
    assert frame.language.tolist() == ['english', 'english']
    assert info['excluded']['empty_text'] == 1


def test_threshold_tradeoff_and_single_class_slices():
    y = [0, 0, 0, 0, 1, 1, 1]
    p = [.1, .2, .5, .55, .4, .45, .9]
    balanced = pick_threshold(y, p, 'balanced')
    precise = pick_threshold(y, p, 'precision_first')
    assert metrics(y, p, balanced)['recall'] > metrics(y, p, precise)['recall']
    with pytest.raises(ValueError):
        pick_threshold(y, p, 'unknown')
    frame = pd.DataFrame({'comment': ['hello', 'idiot'], 'label': [0, 1], 'language': ['a', 'b']})
    result = sliced_metrics(frame, [.1, .9], .5)
    assert result['language']['a']['roc_auc'] is None
    assert result['overall']['recall'] == 1


class FakeTransformer:
    def predict_details(self, texts):
        return np.array([.1, .9, .1][:len(texts)]), [False, False, True][:len(texts)]


def test_transformer_dispatch_truncation_and_diagnostic_overlap():
    bundle = {'backend': 'transformer', 'model': FakeTransformer(), 'threshold': .5,
              'report': {'selected_model': 'test'}}
    cases = [{'comment': 'hello', 'label': 0, 'language': 'english'},
             {'comment': 'abuse', 'label': 1, 'language': 'english'},
             {'comment': 'long', 'label': None, 'language': 'english'}]
    rows = predict(bundle, [c['comment'] for c in cases])
    assert rows[2]['truncated'] and rows[2]['uncertain']
    assert all(row['evidence'] == [] for row in rows)
    report = evaluate_cases(bundle, cases, {group_key('hello')})
    assert report['counts'] == {'total': 3, 'scored': 2, 'failed': 0, 'training_overlap': 1}
    assert report['results'][2]['correct'] is None


def test_temperature_and_incomplete_export(tmp_path):
    temp = fit_temperature([10, 10, -10, -10], [1, 0, 0, 1])
    assert temp > 1
    (tmp_path / 'transformer').mkdir()
    assert load_bundle(tmp_path) is None
    with pytest.raises(ValueError, match='No exported transformer'):
        load_bundle(tmp_path, 'transformer')


def test_preparation_maps_davidson_ids_and_preserves_group_holdouts(tmp_path):
    from toxicity.prepare import prepare
    from toxicity.model import partition
    for split, size in [('train', 80), ('dev', 40), ('test', 40)]:
        rows = [[f'tamil corpus {split} class {label} serial {i}', label_name]
                for label, label_name in [(0, 'Not_offensive'), (1, 'Offensive_Untargetede')]
                for i in range(size)]
        pd.DataFrame(rows).to_csv(tmp_path / f'tamil_offensive_full_{split}.csv', sep='\t', header=False, index=False)
    english = pd.DataFrame([{'tweet': f'english corpus class {label} serial {i}', 'class': label}
                            for label in [0, 1, 2] for i in range(100)])
    english.to_csv(tmp_path / 'labeled_data.csv', index=False)
    frame, info = prepare(tmp_path)
    english = frame[frame.source.eq('davidson')]
    assert english[english.original_label.eq('2')].label.eq(0).all()
    assert english[english.original_label.isin(['0', '1'])].label.eq(1).all()
    assert set(english.language) == {'english'}
    parts, _ = partition(frame)
    for name, part in parts.items():
        for other, other_part in parts.items():
            if name != other:
                assert not set(part.group) & set(other_part.group)
    assert info['saved_rows'] == len(frame)


def test_real_tiny_transformer_train_export_reload_and_api(tmp_path):
    # A tiny randomly initialized BERT tests the real workflow offline, not accuracy.
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from toxicity.train_transformer import train
    base = tmp_path / 'tiny'
    base.mkdir()
    (base / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nfriend\nidiot\nexample\n', encoding='utf-8')
    tokenizer = BertTokenizerFast(vocab_file=str(base / 'vocab.txt'), do_lower_case=False)
    tokenizer.save_pretrained(base)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=24, max_position_embeddings=64, num_labels=2)
    BertForSequenceClassification(config).save_pretrained(base, safe_serialization=True)
    frame = pd.DataFrame([{'comment': f'{phrase} example {i}', 'label': label, 'source': 'tiny', 'language': 'english'}
                          for label, phrase in [(0, 'hello friend'), (1, 'idiot')]
                          for i in range(100)])
    data = tmp_path / 'tiny.csv'
    frame.to_csv(data, index=False)
    runtime = tmp_path / 'runtime'
    report = train(data, runtime / 'models', str(base), epochs=1, batch_size=16,
                   accumulation=3, max_length=32, device='cpu', progress=lambda _: None)
    assert report['best_epoch'] == 1
    bundle = load_bundle(runtime / 'models')
    results = predict(bundle, ['hello friend', 'idiot ' * 100])
    assert all(0 <= row['probability'] <= 1 for row in results)
    assert results[1]['truncated']
    assert results[1]['uncertain']
    assert results[0]['evidence'] == []
    from fastapi.testclient import TestClient
    from toxicity.server import create_app
    client = TestClient(create_app(runtime))
    assert client.get('/api/status').json()['backend'] == 'transformer'
    assert client.post('/api/analyze', json={'comments': ['hello']}).status_code == 200
    assert client.get('/api/export/report').json()['backend'] == 'transformer'
    assert client.post('/api/train', json={}).status_code == 409
    with pytest.raises(ValueError, match='empty output'):
        train(data, runtime / 'models', str(base), epochs=1)
