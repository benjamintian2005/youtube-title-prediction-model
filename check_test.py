import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from train import build_features, load_benchmark


def main():
    model = joblib.load(config.MODEL_PATH)
    raw_df = load_benchmark()
    X, y, split = build_features(raw_df)

    test_mask = (split == "test").to_numpy()
    X_test, y_test = X[test_mask], y[test_mask]

    y_pred = model.predict(X_test)
    log_mse = mean_squared_error(y_test, y_pred)
    log_r2 = r2_score(y_test, y_pred)

    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)
    real_mae = mean_absolute_error(y_test_real, y_pred_real)
    mdape = float(np.median(np.abs((y_test_real - y_pred_real) / np.maximum(y_test_real, 1.0))) * 100)

    print(f"Test set (held out, {config.MODEL_PATH}): n={len(X_test)}")
    print(f"  log_mse: {log_mse:.4f}")
    print(f"  log_r2:  {log_r2:.4f}")
    print(f"  real_mae: {real_mae:.1f}")
    print(f"  mdape:   {mdape:.1f}%")


if __name__ == "__main__":
    main()
