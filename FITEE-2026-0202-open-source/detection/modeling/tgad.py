"""Fit the frozen TGAD LightGBM route and predict user-supplied feature tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', required=True, type=Path)
    parser.add_argument('--predict-csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='A NEW output directory.')
    parser.add_argument('--evaluate', action='store_true', help='Read evaluation labels after training; never tune with them.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output must be a new directory.')
    import numpy as np
    import pandas as pd
    import selected_model as common

    config_path = Path(__file__).resolve().parents[1] / 'configs' / 'tgad_selected.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['selected_model_spec']['model'] != 'lgbm' or config['postprocess'] != {'name':'none'}:
        raise ValueError('Expected the frozen LightGBM/no-postprocess configuration.')
    train = pd.read_csv(args.train_csv, encoding='utf-8-sig', low_memory=False)
    predict = pd.read_csv(args.predict_csv, encoding='utf-8-sig', low_memory=False)
    labels = pd.to_numeric(train['gt_abnormal'], errors='raise')
    if labels.isna().any() or set(labels.unique()) != {0,1}:
        raise ValueError('Training gt_abnormal must contain both binary classes and no missing labels.')
    if args.evaluate:
        predict = common.filter_test_geo_included(predict)
        if predict.empty:
            raise ValueError('No present-pin evaluation rows remain.')
    columns = common.source_cols_from_config(config)
    scores = common.fit_full_scores(train,predict,labels.astype(int).to_numpy(),columns,'lgbm',params=config['selected_model_spec']['params'])
    verdict = (scores >= float(config['threshold'])).astype(int)
    metadata = [c for c in ['sample_uid','sample_id','stable_id'] if c in predict]
    results = predict[metadata].copy()
    results['score'] = scores
    results['pred_abnormal'] = verdict
    report = {'selected_label':config['selected_label'],'threshold':config['threshold'],'postprocess':'none','row_count':len(predict)}
    if args.evaluate:
        truth = pd.to_numeric(predict['gt_abnormal'], errors='raise').to_numpy()
        if not set(np.unique(truth)).issubset({0,1}):
            raise ValueError('Evaluation gt_abnormal must be binary.')
        report['metrics'] = common.metrics_from_pred(truth,scores,verdict)
    args.output.mkdir(parents=True)
    results.to_csv(args.output/'predictions.csv',index=False,encoding='utf-8')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
