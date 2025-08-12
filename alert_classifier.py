# alert_classifier.py
import os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier

MODEL_PATH = os.environ.get("IDS_MODEL", "/home/kali/Desktop/ids_model.json")
EVE_LOG    = os.environ.get("EVE_LOG", "/var/log/suricata/eve.json")
OUT_CSV    = os.environ.get("OUT_CSV", "classified_alerts.csv")

def load_model_json(path: str) -> XGBClassifier:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    m = XGBClassifier()
    m.load_model(str(p))
    return m

def parse_alerts(log_file: str) -> pd.DataFrame:
    rows = []
    proto_map = {"TCP":1, "UDP":2, "ICMP":3, "ICMPV6":3}
    with open(log_file, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event_type") != "alert":
                continue

            alert = rec.get("alert", {}) or {}
            proto_raw = (rec.get("proto") or rec.get("protocol") or "").upper()
            severity = alert.get("severity", 0)

            rows.append({
                "src_ip":   rec.get("src_ip", ""),
                "dest_ip":  rec.get("dest_ip", ""),
                "src_port": rec.get("src_port", 0),
                "dest_port":rec.get("dest_port", 0),
                "proto_raw": proto_raw,
                "proto":    proto_map.get(proto_raw, 0),
                "severity": int(severity) if str(severity).isdigit() else 0,
                "sig_id":   alert.get("signature_id", 0),
                "sig":      alert.get("signature", ""),
            })
    return pd.DataFrame(rows)

def main():
    print(f"[INFO] Model: {MODEL_PATH}")
    print(f"[INFO] EVE  : {EVE_LOG}")

    df = parse_alerts(EVE_LOG)
    if df.empty:
        print("[INFO] No alert events found in eve.json. Nothing to classify.")
        sys.exit(0)

    # base features we can extract now
    base_cols = ["severity", "proto"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0
    X_base = df[base_cols].to_numpy(dtype=np.float32)

    # load model and get expected feature count
    model = load_model_json(MODEL_PATH)
    if not hasattr(model, "n_features_in_"):
        print("[FATAL] Model missing n_features_in_. Cannot pad safely.")
        sys.exit(1)
    exp = int(model.n_features_in_)
    got = X_base.shape[1]

    # pad or trim to expected size
    if got < exp:
        pad = np.zeros((X_base.shape[0], exp - got), dtype=np.float32)
        X = np.concatenate([X_base, pad], axis=1)
        print(f"[INFO] Padded features from {got} -> {exp}")
    elif got > exp:
        X = X_base[:, :exp]
        print(f"[WARN] Trimmed features from {got} -> {exp}")
    else:
        X = X_base

    # predict
    try:
        y = model.predict(X)
    except Exception as e:
        print(f"[FATAL] Prediction failed: {e}")
        sys.exit(1)

    df["prediction"] = y
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote {len(df)} rows -> {OUT_CSV}")
    print(df[["src_ip","dest_ip","proto_raw","severity","prediction"]].head().to_string(index=False))

if __name__ == "__main__":
    main()
