import os
import numpy as np
from torch.utils.data import Dataset


def parse_pdb_to_pointcloud(pdb_file, antibody_chains=None, center=True):
    """
    Parse a PDB file of an antibody-antigen complex and return atom coordinates and features.
    - pdb_file: Path to PDB file.
    - antibody_chains: Optional iterable of chain IDs that correspond to the antibody.
    - center: If True, center the coordinates to the origin.
    Returns: coords (NumPy array of shape [N, 3]), features (NumPy array of shape [N, F]).
    """
    coords = []
    features = []
    # Define one-hot encoding for common atom elements (C, N, O, S, others)
    element_onehot_map = {
        'C': [1, 0, 0, 0, 0],
        'N': [0, 1, 0, 0, 0],
        'O': [0, 0, 1, 0, 0],
        'S': [0, 0, 0, 1, 0]
        # 'others' will be [0,0,0,0,1]
    }
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Skip water molecules for efficiency
                res_name = line[17:20].strip()
                if res_name == "HOH":
                    continue
                # Parse coordinates (PDB columns for x,y,z are 30-38, 38-46, 46-54)
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue  # skip lines that don't have proper coordinates
                coords.append([x, y, z])
                # Determine element type: use element column (76-78) if present, otherwise derive from atom name
                element_sym = line[76:78].strip()
                if element_sym == "":  
                    atom_name = line[12:16].strip()
                    # e.g., atom_name "CA" -> element "C", "NZ" -> "N", etc.
                    element_sym = ''.join([ch for ch in atom_name if ch.isalpha()])[0]  # first letter of atom name
                element_sym = element_sym.capitalize()
                if element_sym in element_onehot_map:
                    elem_one_hot = element_onehot_map[element_sym]
                else:
                    # For uncommon elements, use "others" category
                    elem_one_hot = [0, 0, 0, 0, 1]
                # Optional binary feature for antibody chain membership
                chain_id = line[21:22].strip()  # chain ID is usually column 22
                if antibody_chains is not None:
                    is_antibody = 1 if chain_id in antibody_chains else 0
                    atom_features = elem_one_hot + [is_antibody]
                else:
                    atom_features = elem_one_hot  # no chain info included
                features.append(atom_features)
    coords = np.array(coords, dtype=np.float32)
    features = np.array(features, dtype=np.float32)
    # Center the coordinates (subtract mean) if requested
    if center and len(coords) > 0:
        centroid = coords.mean(axis=0)
        coords = coords - centroid
    return coords, features


class AntibodyAntigenDataset(Dataset):
    """
    PyTorch Dataset for antibody-antigen complexes. Each item is a point cloud with label.
    """
    def __init__(self, pdb_files, labels, antibody_chains_map=None, center=True):
        """
        pdb_files: list of file paths for PDB structures.
        labels: list or array of binary labels (0 or 1) indicating neutralization.
        antibody_chains_map: optional dict mapping pdb_file -> list of antibody chain IDs.
        center: whether to center coordinates.
        """
        assert len(pdb_files) == len(labels), "Files and labels must have same length."
        self.data = []
        self.feature_dim = None
        for i, pdb_path in enumerate(pdb_files):
            chains = None
            if antibody_chains_map is not None and pdb_path in antibody_chains_map:
                chains = antibody_chains_map[pdb_path]
            coords, feats = parse_pdb_to_pointcloud(pdb_path, antibody_chains=chains, center=center)
            label = float(labels[i])
            # Convert to PyTorch tensors
            import torch
            coords_tensor = torch.tensor(coords, dtype=torch.float)
            feats_tensor = torch.tensor(feats, dtype=torch.float)
            label_tensor = torch.tensor(label, dtype=torch.float)
            # Determine feature dimension (set once)
            if self.feature_dim is None:
                self.feature_dim = feats_tensor.shape[1]
            self.data.append((coords_tensor, feats_tensor, label_tensor))
        # If no data loaded (empty list), feature_dim remains None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_point_cloud(batch):
    """
    Custom collate function to batch variable-size point clouds.
    Pads point clouds with zeros and returns masks for actual points.
    """
    import torch
    # Determine the maximum number of points in any sample in this batch
    max_points = max(item[0].shape[0] for item in batch)
    batch_coords = []
    batch_feats = []
    batch_masks = []
    batch_labels = []
    for coords, feats, label in batch:
        n_points = coords.shape[0]
        # Pad coordinates and features to max_points
        if n_points < max_points:
            pad_points = max_points - n_points
            pad_coords = torch.zeros((pad_points, 3), dtype=torch.float)
            pad_feats = torch.zeros((pad_points, feats.shape[1]), dtype=torch.float)
            coords_pad = torch.cat([coords, pad_coords], dim=0)
            feats_pad = torch.cat([feats, pad_feats], dim=0)
            mask = torch.cat([torch.ones(n_points, dtype=torch.bool), torch.zeros(pad_points, dtype=torch.bool)], dim=0)
        else:
            coords_pad = coords
            feats_pad = feats
            mask = torch.ones(n_points, dtype=torch.bool)
        batch_coords.append(coords_pad)
        batch_feats.append(feats_pad)
        batch_masks.append(mask)
        batch_labels.append(label)
    # Stack all
    batch_coords = torch.stack(batch_coords, dim=0)   # shape: (B, max_points, 3)
    batch_feats = torch.stack(batch_feats, dim=0)     # shape: (B, max_points, F)
    batch_masks = torch.stack(batch_masks, dim=0)     # shape: (B, max_points)
    batch_labels = torch.stack(batch_labels, dim=0)   # shape: (B,)
    return batch_coords, batch_feats, batch_masks, batch_labels
