from flask import Flask, request
import pandas as pd
import os
import requests

app = Flask(__name__)

# LINE Notify
LINE_NOTIFY_TOKEN = os.environ.get("LINE_NOTIFY_TOKEN")

def line_notify(message):
    if LINE_NOTIFY_TOKEN:
        requests.post(
            "https://notify-api.line.me/api/notify",
            headers={"Authorization": f"Bearer {LINE_NOTIFY_TOKEN}"},
            data={"message": message},
        )

# Webhook → CSV POST
@app.route("/", methods=["POST"])
def receive_csv():
    file = request.files.get("csv")
    if not file:
        return "No CSV file uploaded", 400

    try:
        df = pd.read_csv(file)
        summary = df.head().to_string(index=False)
        print(f"✅ CSV received ({len(df)} rows):\n{summary}")
        line_notify(f"📥 OCR CSV received!\n{summary}")
        return "CSV processed", 200
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}", 500

# LINE webhook endpoint (そのまま保持)
@app.route("/callback", methods=["POST"])
def callback():
    return "LINE webhook (not used in this test)", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)