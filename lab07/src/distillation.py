import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=4.0, alpha=0.5):
    """Combine hard-label CE and temperature-scaled teacher-student KL divergence."""
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss

def train_student(student, teacher, train_loader, device, epochs, learning_rate=1e-3,
                  use_distillation=True, temperature=4.0, alpha=0.5):
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)
    student.train()
    if teacher is not None:
        teacher.eval()
    for epoch in range(epochs):
        running_loss, seen = 0.0, 0
        student.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(images)
            if use_distillation and teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = distillation_loss(student_logits, teacher_logits, labels,
                                         temperature, alpha)
            else:
                loss = F.cross_entropy(student_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        print(f"Student epoch {epoch + 1}/{epochs} - loss: {running_loss / max(seen, 1):.4f}")
    return student
