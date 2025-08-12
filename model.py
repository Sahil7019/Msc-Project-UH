# convert_model.py
import sys, os, traceback
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 convert_model.py <input_model.pkl> <output_model.json>")
        sys.exit(2)

    in_path = Path(sys.argv[1]).expanduser().resolve()
    out_path = Path(sys.argv[2]).expanduser().resolve()

    print(f"[INFO] Python: {sys.version}")
    try:
        import xgboost as xgb
        import joblib
    except Exception:
        print("[ERR] Missing deps. In your venv: pip install xgboost joblib")
        sys.exit(1)

    print(f"[INFO] XGBoost: {xgb.__version__}")
    print(f"[INFO] Input : {in_path}  (exists={in_path.exists()})")
    print(f"[INFO] Output: {out_path}")

    if not in_path.exists():
        print("[FATAL] Input model file not found.")
        sys.exit(1)

    try:
        model = joblib.load(in_path)
        print(f"[INFO] Loaded object type: {type(model)}")

        # Case A: sklearn wrapper (XGBClassifier / XGBRFClassifier / XGBRegressor)
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            print("[INFO] Detected sklearn wrapper -> saving booster as JSON")
            booster.save_model(str(out_path))

        # Case B: raw Booster (some people pickle the Booster directly)
        elif isinstance(model, xgb.Booster):
            print("[INFO] Detected raw Booster -> saving as JSON")
            model.save_model(str(out_path))

        else:
            print("[FATAL] Unsupported object type. You likely pickled a non-XGBoost model.")
            sys.exit(1)

        print(f"[OK] Saved JSON model -> {out_path}")

    except Exception as e:
        print("[FATAL] Conversion failed:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
