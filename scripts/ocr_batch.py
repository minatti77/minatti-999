#!/usr/bin/env python3
import argparse, pathlib, re, pandas as pd
from PIL import Image
import pytesseract

def main(pattern: str, out: pathlib.Path):
    rows = []
    for img in pathlib.Path(".").glob(pattern):
        text = pytesseract.image_to_string(Image.open(img), lang="jpn")
        for m in re.finditer(r'(\d{1,2})[^\d]*(\d+\.\d+)', text):
            rows.append({"馬番": int(m.group(1)), "オッズ": float(m.group(2)), "画像": img.name})

    df = (pd.DataFrame(rows)
            .drop_duplicates("馬番")
            .sort_values("オッズ")
            .reset_index(drop=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[✅] Saved {len(df)} rows → {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="Screenshot*.jpg")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("output/ocr.csv"))
    main(**vars(ap.parse_args()))
