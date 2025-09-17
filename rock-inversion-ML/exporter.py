# exporter.py
import pandas as pd

def export_predictions(y_true, y_pred, path):
    """Save predictions to Excel."""
    df = pd.DataFrame({'True': y_true.ravel(), 'Predicted': y_pred.ravel()})
    df.to_excel(path, index=False)
    print(f"[INFO] Predictions saved to {path}")

def export_inversion_results(inversion_results, path):
    rows = []
    for res in inversion_results:
        if res is None or res.get("Best Params") is None:
            continue  # skip if no valid result
        best_params = res["Best Params"]
        loss = res.get("Loss", None)
        target = res.get("Target", None)
        row = list(best_params) + [loss] + list(target)
        rows.append(row)

    if rows:
        columns = [f"Param_{i+1}" for i in range(len(inversion_results[0]['Best Params']))] + ["Loss"] + ["Target"]
        df = pd.DataFrame(rows, columns=columns)
        df.to_excel(path, index=False)
        print(f"[INFO] Inversion results exported to: {path}")
    else:
        print("[INFO] No valid inversion results to export.")

