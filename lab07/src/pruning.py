import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

def _eligible_parameters(model):
    return [(module, "weight") for module in model.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))]

def apply_magnitude_pruning(model, amount, mode="global"):
    """Apply unstructured L1-magnitude pruning globally or independently per layer."""
    parameters = _eligible_parameters(model)
    if amount <= 0:
        return model
    if mode == "global":
        prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)
    elif mode == "layerwise":
        for module, name in parameters:
            prune.l1_unstructured(module, name=name, amount=amount)
    else:
        raise ValueError("mode must be 'global' or 'layerwise'")
    return model

def apply_structured_filter_pruning(model, amount):
    """Mask complete convolutional output filters using their L2 norms."""
    if amount <= 0:
        return model
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    return model

def enforce_pruning_masks(model):
    """Materialize pruning reparameterizations while retaining zeroed weights."""
    for module in model.modules():
        if hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    return model

def sparsity_by_layer(model):
    report = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.detach()
            report[name] = 1.0 - torch.count_nonzero(weight).item() / weight.numel()
    return report

def overall_sparsity(model):
    zeros, total = 0, 0
    for module, _ in _eligible_parameters(model):
        weight = module.weight.detach()
        zeros += (weight == 0).sum().item()
        total += weight.numel()
    return zeros / max(total, 1)
