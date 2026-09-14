import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def validate_gold(model, loader, device, target_names):
    """Evaluate a model using per-target and macro ROC AUC."""
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            sagittal = batch["sagittal"].to(device, non_blocking=True)
            coronal = batch["coronal"].to(device, non_blocking=True)
            axial = batch["axial"].to(device, non_blocking=True)
            targets = batch["targets"].cpu().numpy()

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(sagittal, coronal, axial)

            probs = torch.sigmoid(logits).float().cpu().numpy()
            all_targets.append(targets)
            all_probs.append(probs)

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_probs, axis=0)

    aucs = {
        name: roc_auc_score(y_true[:, i], y_pred[:, i])
        for i, name in enumerate(target_names)
    }
    macro_auc = float(np.mean(list(aucs.values())))

    return macro_auc, aucs, y_true, y_pred
