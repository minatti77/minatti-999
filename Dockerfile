FROM python:3.11-slim

WORKDIR /app

# ── システム依存パッケージ（Tesseract OCR + 日本語対応）──
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依存ライブラリ ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── アプリ本体をコピー ──
COPY . .

# ── アップロード・出力ディレクトリを作成 ──
RUN mkdir -p uploads output

ENV PORT=10000
EXPOSE 10000

CMD ["python", "app.py"]
