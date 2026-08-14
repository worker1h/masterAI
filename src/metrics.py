def binary_metrics(tp, fp, fn, tn, eps=1e-9):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return {"iou": iou, "dice": dice, "f1": dice, "precision": precision, "recall": recall, "accuracy": accuracy}

