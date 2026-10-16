import torch 
from torch import Tensor 
from torchmetrics.functional.classification import multiclass_f1_score, binary_average_precision
# from torchmetrics.functional.clustering import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compute_f1(
    input: Tensor,
    target: Tensor,
    average: str,
) -> float:
    batch_size, num_classes = input.shape  
    assert target.shape == (batch_size,)
    assert int(target.max() + 1) == num_classes 
    assert input.dtype == torch.float and target.dtype == torch.long

    f1 = multiclass_f1_score(
        preds = input,
        target = target,
        num_classes = num_classes,
        average = average,
    )

    return float(f1)


def compute_micro_f1(
    input: Tensor,
    target: Tensor,
) -> float:
    return compute_f1(
        input = input, 
        target = target, 
        average = 'micro',
    )


def compute_macro_f1(
    input: Tensor,
    target: Tensor,
) -> float:
    return compute_f1(
        input = input, 
        target = target, 
        average = 'macro',
    )


def compute_ap(
    input: Tensor,
    target: Tensor,
) -> float:
    batch_size, num_classes = input.shape  
    assert target.shape == (batch_size,)
    assert int(target.max() + 1) == num_classes == 2 
    assert input.dtype == torch.float and target.dtype == torch.long

    f1 = binary_average_precision(
        preds = input[:, 1],
        target = target,
    )

    return float(f1)


def compute_nmi(
    input: Tensor,
    target: Tensor,
) -> float:
    batch_size, = input.shape  
    assert target.shape == (batch_size,)
    assert input.dtype == target.dtype == torch.long 

    nmi = normalized_mutual_info_score(
        labels_pred = input.cpu().numpy(),
        labels_true = target.cpu().numpy(),
    )

    return float(nmi)


def compute_ari(
    input: Tensor,
    target: Tensor,
) -> float:
    batch_size, = input.shape  
    assert target.shape == (batch_size,)
    assert input.dtype == target.dtype == torch.long 

    ari = adjusted_rand_score(
        labels_pred = input.cpu().numpy(),
        labels_true = target.cpu().numpy(),
    )

    return float(ari)
