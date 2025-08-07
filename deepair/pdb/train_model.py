from .preprocessing import AntibodyAntigenDataset, collate_point_cloud
from .model import NeutralizationModel
from .training import train_model
import torch

# Example file lists and labels (in practice, populate these with real data paths and labels)
train_files = ["data/complex1.pdb", "data/complex2.pdb", "..."]  # list of training PDB file paths
train_labels = [0, 1, ...]  # binary labels for each training complex (0 = non-neutralizing, 1 = neutralizing)
val_files = ["data/complexA.pdb", "data/complexB.pdb"]         # list of validation PDB file paths
val_labels = [0, 1]                                            # labels for validation complexes

# (Optional) If known, provide a map of PDB file to antibody chain IDs for feature augmentation
antibody_chains_map = {
    # "data/complex1.pdb": ["H", "L"],  # e.g., heavy and light chains labeled H, L in this PDB are antibody
    # "data/complex2.pdb": ["A", "B"],  # example where chains A and B are antibody chains, etc.
}

# Create Dataset and DataLoader for training and validation sets
train_dataset = AntibodyAntigenDataset(train_files, train_labels, antibody_chains_map=antibody_chains_map)
val_dataset = AntibodyAntigenDataset(val_files, val_labels, antibody_chains_map=antibody_chains_map)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_point_cloud)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_point_cloud)

# Initialize the model with SE(3) Transformer backbone
model = NeutralizationModel(
    backbone='se3', 
    num_input_features=train_dataset.feature_dim,   # number of per-atom features (e.g., 6 if element+chain flag)
    backbone_params={
        "dim": 64,      # embed dimension for SE3 Transformer
        "depth": 4,     # number of transformer layers
        "heads": 4,     # number of attention heads
        # "num_degrees": 4, etc. (we can tune other SE3Transformer params as needed)
    }
)
# Alternatively, for a baseline model:
# model = NeutralizationModel(backbone='mlp', num_input_features=train_dataset.feature_dim,
#                             backbone_params={"hidden_dim": 128, "dropout": 0.1})

# Set up training configuration
config = {
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,         # L2 regularization
    "scheduler": "step", 
    "step_size": 10,
    "gamma": 0.5,                 # halve the LR every 10 epochs
    "early_stop_patience": 5,     # stop if no improvement for 5 epochs
    "pos_weight": 2.0,            # e.g., if neutralizing examples are rarer, give them higher weight
    "label_smoothing": 0.1        # soften labels from {0,1} to {0.1,0.9}
}

# Train the model
best_model, best_metrics = train_model(model, train_loader, val_loader, config)

# After training, best_model contains the weights of the best epoch (highest val AUC).
# We can evaluate on the validation set or test set to see final performance:
print("Best Validation AUC: {:.3f}".format(best_metrics["auc"]))
print("Best Validation Accuracy: {:.3f}".format(best_metrics["accuracy"]))
print("Best Validation Precision: {:.3f}".format(best_metrics["precision"]))
print("Best Validation Recall: {:.3f}".format(best_metrics["recall"]))

# (Optional) Save the trained model
torch.save(best_model.state_dict(), "neutralization_model.pth")
