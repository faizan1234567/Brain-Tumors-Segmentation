import os
import torch
import pandas as pd
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from pathlib import Path

# Save best model
def save_best_model(dir_name, model, name="best_model"):
    """Save best model weights"""
    save_path = os.path.join(dir_name, "best-model", name)
    torch.save(model.state_dict(), f"{save_path}.pth")

# Save checkpoint
def save_checkpoint(dir_name, state, name="checkpoint"):
    """Save checkpoint with each epoch to resume"""
    save_path = os.path.join(dir_name, "checkpoint", name)
    torch.save(state, f"{save_path}.pth.tar")

# Create directories
def create_dirs(dir_name):
    """Create experiment directory storing checkpoint and best weights"""
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "best-model"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "csv"), exist_ok=True)


# Validation function
def validation(model, post_label, post_pred, dice_metric_per_class, epoch_iterator_val, device):
    model.eval()
    class_names = [
        "Spleen", "Right_Kidney", "Left_Kidney", "Gallbladder", "Esophagus", "Liver",
        "Stomach", "Aorta", "IVC", "Portal_and_Splenic_Veins", "Pancreas",
        "Right_adrenal_gland", "Left_adrenal_gland"
    ]
    class_dice_scores = {name: [] for name in class_names}
    mean_dice_val = 0

    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            with torch.cuda.amp.autocast():
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)

            val_labels_list = decollate_batch(val_labels)
            val_outputs_list = decollate_batch(val_outputs)

            for val_label_tensor, val_pred_tensor in zip(val_labels_list, val_outputs_list):
                val_label = post_label(val_label_tensor)
                val_pred = post_pred(val_pred_tensor)
                dice_metric_per_class(y_pred=val_pred[1:], y=val_label[1:])  # Exclude background

            class_dice = dice_metric_per_class.aggregate()
            for i, name in enumerate(class_names):
                class_dice_scores[name].append(class_dice[i].item())

            dice_metric_per_class.reset()

        mean_dice_val = sum([sum(scores) / len(scores) for scores in class_dice_scores.values()]) / len(class_names)

    return mean_dice_val, {name: sum(scores) / len(scores) for name, scores in class_dice_scores.items()}

# Training function
def train(
    model, loss_function, optimizer, scaler, train_loader, val_loader,
    max_iterations, eval_num, exp_name,
    post_label, post_pred, dice_metric_per_class, device
):
    create_dirs(exp_name)
    model.train()
    epoch_loss_values = []
    metric_values = []
    class_dice_values = []
    checkpoint_interval = 0

    # CSV structure
    columns = ["Epoch", "Training_Loss", "Validation_Loss", "Mean_Dice"] + [
        f"Dice_{name}" for name in [
            "Spleen", "Right_Kidney", "Left_Kidney", "Gallbladder", "Esophagus", "Liver",
            "Stomach", "Aorta", "IVC", "Portal_and_Splenic_Veins", "Pancreas",
            "Right_adrenal_gland", "Left_adrenal_gland"
        ]
    ]
    csv_path = Path(exp_name) / "csv" / f"{exp_name}_data.csv"
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

    for epoch in range(max_iterations):
        model.train()
        epoch_loss = 0
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_iterations}", dynamic_ncols=True)

        for batch in epoch_iterator:
            x, y = (batch["image"].to(device), batch["label"].to(device))
            with torch.cuda.amp.autocast():
                logit_map = model(x)
                loss = loss_function(logit_map, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        epoch_loss_values.append(epoch_loss)

        if (epoch + 1) % eval_num == 0 or (epoch + 1) == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}", dynamic_ncols=True)
            mean_dice, class_dice_scores = validation(model, post_label, post_pred, dice_metric_per_class, epoch_iterator_val, 
                                                      device=device)

            metric_values.append(mean_dice)
            class_dice_values.append(class_dice_scores)

            row = {
                "Epoch": epoch + 1,
                "Training_Loss": epoch_loss,
                "Validation_Loss": None,  # Placeholder if validation loss is needed
                "Mean_Dice": mean_dice,
            }
            row.update({f"Dice_{name}": class_dice_scores[name] for name in class_dice_scores})

            pd.DataFrame([row]).to_csv(csv_path, mode="a", header=False, index=False)

            # Save best model
            if mean_dice > dice_val_best:
                dice_val_best = mean_dice
                save_best_model(exp_name, model)

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "dice_val_best": dice_val_best,
            }
            save_checkpoint(exp_name, checkpoint_state)

    return dice_val_best
