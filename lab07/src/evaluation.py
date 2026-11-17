import io
import time
import torch

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"accuracy": correct / max(total, 1), "loss": loss_sum / max(total, 1)}


def measure_latency_ms(model, device, warmup=30, iterations=100):
    """Measure batch-1 inference latency; synchronize CUDA around timed regions."""
    model.eval()
    sample = torch.randn(1, 1, 28, 28, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iterations


def serialized_size_mb(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 ** 2)


def sparse_coo_size_mb(model):
    """Estimate COO storage for nonzero parameters (values + 64-bit indices)."""
    total_bytes = 0
    for parameter in model.parameters():
        tensor = parameter.detach().cpu()
        nonzero = torch.count_nonzero(tensor).item()
        total_bytes += nonzero * (tensor.element_size() + 8 * tensor.ndim)
    return total_bytes / (1024 ** 2)
