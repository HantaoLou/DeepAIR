import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


def train_model(model, train_loader, val_loader, config):
    """
    Train the model with given data loaders and configuration.
    Returns the best model (with highest validation AUC) and a dictionary of its best metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Optimizer (Adam) with optional L2 weight decay for regularization
    learning_rate = config.get('learning_rate', 1e-3)
    weight_decay = config.get('weight_decay', 0.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Learning rate scheduler (optional)
    scheduler = None
    if config.get('scheduler', None) == 'step':
        step_size = config.get('step_size', 10)
        gamma = config.get('gamma', 0.5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif config.get('scheduler', None) == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    # Loss function: Weighted BCE with optional label smoothing
    pos_weight_val = config.get('pos_weight', None)
    if pos_weight_val is not None:
        pos_weight = torch.tensor(pos_weight_val, dtype=torch.float, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    label_smoothing = config.get('label_smoothing', 0.0)
    # Early stopping parameters
    patience = config.get('early_stop_patience', 10)
    best_auc = -float('inf')
    best_model_state = None
    epochs = config.get('epochs', 50)
    patience_counter = 0

    for epoch in range(1, epochs+1):
        # ---- Training loop ----
        model.train()
        total_loss = 0.0
        for coords, feats, mask, labels in train_loader:
            coords = coords.to(device)
            feats = feats.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            logits = model(coords, feats, mask)  # raw scores
            # Apply optional label smoothing to the targets
            if label_smoothing and label_smoothing > 0:
                # Mix the labels with its opposite label by 'label_smoothing' fraction
                labels_smoothed = labels * (1.0 - label_smoothing) + (1.0 - labels) * label_smoothing
            else:
                labels_smoothed = labels
            # Compute loss (BCE with logits)
            loss = criterion(logits, labels_smoothed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)
        # ---- Validation loop ----
        model.eval()
        val_losses = 0.0
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for coords, feats, mask, labels in val_loader:
                coords = coords.to(device); feats = feats.to(device); mask = mask.to(device); labels = labels.to(device)
                logits = model(coords, feats, mask)
                # Use the same label smoothing for validation loss calculation (not strictly necessary)
                if label_smoothing and label_smoothing > 0:
                    labels_smoothed = labels * (1.0 - label_smoothing) + (1.0 - labels) * label_smoothing
                else:
                    labels_smoothed = labels
                loss = criterion(logits, labels_smoothed)
                val_losses += loss.item() * labels.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        # Concatenate all predictions and labels
        all_logits = torch.cat(all_logits)         # shape (num_val_samples,)
        all_labels = torch.cat(all_labels)         # shape (num_val_samples,)
        avg_val_loss = val_losses / len(val_loader.dataset)
        # Compute metrics
        # Convert logits to probabilities
        all_probs = torch.sigmoid(all_logits).numpy()
        y_true = all_labels.numpy()
        # Binary predictions for threshold 0.5
        y_pred = (all_probs >= 0.5).astype(int)
        # Accuracy, Precision, Recall
        val_accuracy = accuracy_score(y_true, y_pred)
        # Handle case where there might be no positive predictions or labels to avoid zero-division
        if y_pred.sum() == 0 and y_true.sum() == 0:
            # degenerate case: no positives in data or predicted, define precision=recall=1.0
            val_precision = 1.0
            val_recall = 1.0
        else:
            val_precision = precision_score(y_true, y_pred, zero_division=0)
            val_recall = recall_score(y_true, y_pred, zero_division=0)
        # AUC (Only if both classes present; otherwise, we skip to avoid error)
        try:
            val_auc = roc_auc_score(y_true, all_probs)
        except ValueError:
            val_auc = float('nan')
        # Print epoch summary
        print(f"Epoch {epoch}: Train loss = {avg_train_loss:.4f}, Val loss = {avg_val_loss:.4f}, "
              f"Val AUC = {val_auc:.3f}, Val Acc = {val_accuracy:.3f}, "
              f"Val Prec = {val_precision:.3f}, Val Rec = {val_recall:.3f}")
        # Learning rate scheduler step
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        # Early stopping check on AUC (or could use val_loss)
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    # Load best model weights before returning (best on validation AUC)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    best_metrics = {"auc": best_auc, "accuracy": val_accuracy, "precision": val_precision, "recall": val_recall}
    return model, best_metrics
