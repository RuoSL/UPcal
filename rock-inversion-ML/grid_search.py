# === grid_search.py ===
import numpy as np
import itertools
import pandas as pd
from itertools import product
from tqdm import tqdm
from sklearn.metrics import r2_score


def generate_grid(param_ranges):
    grid = list(itertools.product(*param_ranges))
    print(f"[INFO] Generated grid shape: {len(grid)} rows")
    return np.array(grid)

def multi_stage_inversion(model, scaler_X, scaler_Y, target_df, param_ranges_coarse, steps_fine, tolerance):
    result_data = []
    target_df = target_df.dropna()

    # === Generate coarse search grid ===
    all_combinations_coarse = np.array(list(product(*param_ranges_coarse)))
    all_combinations_coarse_standardized = scaler_X.transform(all_combinations_coarse)

    for idx, (index, row) in tqdm(enumerate(target_df.iterrows()), total=len(target_df), desc="Inversion Rows"):
        target_output = row.values.reshape(1, -1)
        loss_list = []

        # === Coarse grid search ===
        for inputs_std in all_combinations_coarse_standardized:
            pred_std = model.predict(inputs_std.reshape(1, -1))[0]
            pred = scaler_Y.inverse_transform(pred_std.reshape(1, -1))[0]
            loss = np.sum((pred - target_output[0]) ** 2)
            loss_list.append({
                'inputs': scaler_X.inverse_transform([inputs_std])[0],
                'predicted_output': pred,
                'loss': loss
            })

        best = min(loss_list, key=lambda x: x['loss'])
        best_inputs = best['inputs']

        # === Multi-stage refined search ===
        for step in steps_fine:
            fine_ranges = []
            for i, val in enumerate(best_inputs):
                low = max(val - step*10, param_ranges_coarse[i][0])
                high = min(val + step*10, param_ranges_coarse[i][-1])
                fine_ranges.append(np.arange(low, high, step))

            fine_grid = np.array(list(product(*fine_ranges)))
            fine_grid_std = scaler_X.transform(fine_grid)

            loss_list_fine = []
            for inputs_std in fine_grid_std:
                pred_std = model.predict(inputs_std.reshape(1, -1))[0]
                pred = scaler_Y.inverse_transform(pred_std.reshape(1, -1))[0]
                loss = np.sum((pred - target_output[0]) ** 2)
                loss_list_fine.append({
                    'inputs': scaler_X.inverse_transform([inputs_std])[0],
                    'predicted_output': pred,
                    'loss': loss
                })

            best = min(loss_list_fine, key=lambda x: x['loss'])
            best_inputs = best['inputs']

            if best['loss'] <= tolerance:
                break

        r2 = r2_score(target_output[0], best['predicted_output'])
        row_result = {'Row': index, 'Loss': best['loss'], 'R2': r2}
        row_result.update({f"Target_{col}": val for col, val in zip(target_df.columns, target_output[0])})
        row_result.update({f"Predicted_{col}": val for col, val in zip(target_df.columns, best['predicted_output'])})
        for i, inp in enumerate(best_inputs):
            row_result[f"Input{i+1}"] = inp

        result_data.append(row_result)

    return result_data