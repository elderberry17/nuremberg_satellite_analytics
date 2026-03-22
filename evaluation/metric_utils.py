import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def per_class_mae(y_true, y_pred, class_names):
    return {
        class_names[i]: float(np.mean(np.abs(y_true[:, i] - y_pred[:, i])))
        for i in range(len(class_names))
    }


def mae_macro(y_true, y_pred):
    per_class = np.mean(np.abs(y_true - y_pred), axis=0)
    return float(np.mean(per_class))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_macro(y_true, y_pred):
    per_class = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return float(np.mean(per_class))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return float(1 - ss_res / ss_tot)


def r2_macro(y_true, y_pred):
    r2s = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)

        r2s.append(1 - ss_res / ss_tot if ss_tot != 0 else 0.0)

    return float(np.mean(r2s))


def kl_divergence(y_true, y_pred, eps=1e-10):
    y_true = np.clip(y_true, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)
    return np.mean(np.sum(y_true * np.log(y_true / y_pred), axis=1))


def evaluate_composition_metrics(y_true, y_pred, class_names):
    results = {}

    # --- overall (macro) ---
    results["overall"] = {
        "mae_macro": mae_macro(y_true, y_pred),
        "rmse_macro": rmse_macro(y_true, y_pred),
        "r2_macro": r2_macro(y_true, y_pred),
        "kl": kl_divergence(y_true, y_pred),
    }

    # --- per-class ---
    results["per_class"] = {}

    for i, name in enumerate(class_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mae_i = float(np.mean(np.abs(yt - yp)))
        rmse_i = float(np.sqrt(np.mean((yt - yp) ** 2)))

        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2_i = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

        results["per_class"][name] = {
            "mae": mae_i,
            "rmse": rmse_i,
            "r2": r2_i,
        }

    return results


def change_mask(y, threshold=1e-3):
    return np.abs(y) > threshold


def mcr(y_true, y_pred, threshold=1e-3):
    true_change = change_mask(y_true, threshold)
    pred_change = change_mask(y_pred, threshold)

    tp = np.logical_and(true_change, pred_change).sum()
    fn = np.logical_and(true_change, ~pred_change).sum()

    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def fcr(y_true, y_pred, threshold=1e-3):
    true_change = change_mask(y_true, threshold).sum()
    pred_change = change_mask(y_pred, threshold).sum()

    return float(pred_change / true_change) if true_change > 0 else 0.0


def direction_accuracy(y_true, y_pred):
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)

    return float(np.mean(true_sign == pred_sign))


def per_class_direction_accuracy(y_true, y_pred, class_names):
    results = {}
    for i, name in enumerate(class_names):
        true_sign = np.sign(y_true[:, i])
        pred_sign = np.sign(y_pred[:, i])

        results[name] = float(np.mean(true_sign == pred_sign))

    return results


def evaluate_change_metrics(y_true, y_pred, class_names, threshold=1e-3):
    results = {}

    # --- overall ---
    results["overall"] = {
        "mcr": mcr(
            y_true, y_pred, threshold
        ),  # missed change rate (recall-like)
        "fcr": fcr(y_true, y_pred, threshold),  # false change ratio
        "direction_accuracy": direction_accuracy(y_true, y_pred),
    }

    # --- per-class ---
    results["per_class"] = {}

    for i, name in enumerate(class_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        results["per_class"][name] = {
            "mcr": mcr(yt, yp, threshold),
            "fcr": fcr(yt, yp, threshold),
            "direction_accuracy": float(np.mean(np.sign(yt) == np.sign(yp))),
        }

    return results
