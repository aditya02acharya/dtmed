import torch


def add_padding(feature: torch.Tensor = None, padding_len: int = 0, dim: int = 0) -> torch.Tensor:
    """
    Description:
        Utility function to add padding to feature vector of requested size.

    Arguments:
        feature: numpy array to be padded.
        padding_len: integer length of padding size.
        dim: axis to concatenate on.

    Return:
         Tensor object
    """
    return torch.cat([feature, torch.zeros(([padding_len] + list(feature.shape[1:])), dtype=feature.dtype)], dim=dim)
