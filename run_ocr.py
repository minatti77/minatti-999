#!/usr/bin/env python3
import argparse, zipfile, pathlib, tempfile, yaml, pandas as pd, duckdb, cv2, pytesseract
from datetime import datetime
from PIL import Image

def preprocess(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th

def ocr_img(img, psm, oem):
    cfg = f'--psm {psm} --oem {oem}'
    return pytesseract.image_to_string(img, lang='jpn', config=cfg).strip()

def run(cfg):
    in_path = pathlib.Path(cfg['input_path'])
    files = []
    if in_path.suffix == '.zip':
        with zipfile.ZipFile(in_path) as z:
            with tempfile.TemporaryDirectory() as td:
                z.extractall(td)
                files = list(pathlib.Path(td).rglob('*.png'))
    elif in_path.is_dir():
        files = list(in_path.rglob('*.png'))
    else:
        files = [in_path]

    rec = []
    for f in files:
        txt = ocr_img(preprocess(f), cfg['psm'], cfg['oem'])
        rec.append({'file': f.name, 'text': txt})

    df = pd.DataFrame(rec)
    out = pathlib.Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv = out / f'ocr_{ts}.csv'
    df.to_csv(csv, index=False)

    if cfg.get('duckdb'):
        con = duckdb.connect(cfg['duckdb'])
        con.execute('CREATE TABLE IF NOT EXISTS ocr_results AS SELECT * FROM df LIMIT 0;')
        con.execute('INSERT INTO ocr_results SELECT * FROM df;')
        con.close()

    print('[✓] OCR finished', csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='ocr_config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)
