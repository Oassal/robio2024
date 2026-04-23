import torch
from tqdm import tqdm

def train_epoch(teacher_model, 
                student_model, 
                device, 
                train_loader, 
                optimizer, 
                criterion, 
                alpha, 
                epoch, 
                NUM_EPOCHS):
    teacher_model.eval()
    student_model.train()
    train_loss = 0.0
    with tqdm(total=len(train_loader), desc=f"Epoch: {epoch + 1}/{NUM_EPOCHS}", leave = False) as batch_bar:
        for images, masks in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            # single forward pass: (student_output, decoded, z_mean, z_log_var)
            student_output, decoded, z_mean, z_log_var = student_model(images)

            # Teacher output
            with torch.no_grad():
                teacher_output = teacher_model(images)
            
            # KD loss
            kd_loss = criterion(student_output, teacher_output)

            # KL divergence loss
            kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), dim=1)
            loss = alpha * kd_loss + kl_div.mean()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            batch_bar.set_postfix(loss=loss.item())
            batch_bar.update(1)
    return train_loss
