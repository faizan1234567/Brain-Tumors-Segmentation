"""
performance measurment metrics 
dice coefficient score and
IOU score
"""
import numpy as np
import torch

def dice_metric(probabilites: torch.Tensor,
                labels: torch.Tensor,
                threshold: float = 0.5,
                eps: float = 1e-9)->np.ndarray:
    """
    calculate dice score given model output probablities and ground truth labels 

    Parameters
    ----------
    probablitites: torch.Tensor
    labels: torch.Tensor
    threshold: float = 0.5 (Default)
    eps: float = 1e-9 (Default)
    """
    scores = []
    num = probabilites.shape[0]
    predictions = (probabilites >= threshold).float()
    assert predictions.shape == labels.shape
    for i in range(num):
        prediction = predictions[i]
        label = labels[i]
        intersection = 2.0 * (label * prediction).sum()
        union = label.sum() + prediction.sum()
        if label.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps)/union)
    return np.mean(scores)

def jaccard_metric(probabilites: torch.Tensor,
                labels: torch.Tensor,
                threshold: float = 0.5,
                eps: float = 1e-9)->np.ndarray:
    """
    calculate jaccard cofficient score given model output probablities and ground truth labels 

    Parameters
    ----------
    probablitites: torch.Tensor
    labels: torch.Tensor
    threshold: float = 0.5 (Default)
    eps: float = 1e-9 (Default)
    """
    scores = []
    num = probabilites.shape[0]
    predictions = (probabilites >= threshold).float()
    assert predictions.shape == labels.shape
    for i in range(num):
        prediction = predictions[i]
        label = labels[i]
        intersection = (label * prediction).sum()
        union = label.sum() + prediction.sum() - intersection + eps
        if label.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps)/union)
    return np.mean(scores)

if __name__ == '__main__':
    probs = torch.rand(3, 155, 240, 240)
    gt = torch.rand(3, 155, 240, 240)
    score = dice_metric(probs, gt)
    jaccard_score = jaccard_metric(probs, gt)
    print(f'mean dice score: {score}')
    print(f'jaccard coff: {jaccard_score}')