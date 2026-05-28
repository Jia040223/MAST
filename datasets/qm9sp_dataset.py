"""QM9SP dataset wrapper used by the public MAST release."""

import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from functools import lru_cache
import torch_geometric.data.data

# Force `weights_only=False` for older serialized PyG dataset files.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Call `torch.load` with `weights_only=False` by default."""
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


class QM9SP(QM9_geometric):
    """
    QM9 dataset with UV, IR, and Raman spectra.
    """
    
    def __init__(self, root, transform=None, dataset_arg=None, modalities=('uv', 'ir', 'raman')):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm9_target_dict.values())}.'
        )

        self.label = dataset_arg
        if dataset_arg == "alpha":   # set this value as placeholder during pre-training
            self.label = "isotropic_polarizability"
        elif dataset_arg in ["U", "U0"]:
            self.label = "energy_" + dataset_arg

        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.label]
        
        self.modalities = modalities
        self.spectra_field_names = self._detect_spectra_field_names()

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        # Allowlist PyG data classes for PyTorch 2.6+ loading.
        try:
            import torch.serialization
            from torch_geometric.data.data import Data, DataEdgeAttr
            torch.serialization.add_safe_globals([Data, DataEdgeAttr])
        except:
            pass
        
        super(QM9SP, self).__init__(root, transform=transform)

    @property
    def processed_file_names(self) -> str:
        return "data_with_uv_ir_raman.pt"
    
    def _process(self):
        """Override the loader to keep compatibility with older dataset files."""
        import torch
        f = open(self.processed_paths[0], 'rb')
        try:
            self.data, self.slices = torch.load(f, weights_only=False)
        finally:
            f.close()

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        """Keep only the selected QM9 target."""
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch
    
    def _detect_spectra_field_names(self):
        """
        Detect the spectral field names lazily on the first sample.
        """
        return None
    
    def __getitem__(self, idx):
        """
        Load one sample and normalize the spectral field naming.
        """
        data = super().__getitem__(idx)
        
        if self.spectra_field_names is None:
            self.spectra_field_names = {}
            for mod in self.modalities:
                if hasattr(data, mod):
                    self.spectra_field_names[mod] = mod
                else:
                    candidates = [f'{mod}_spectra', f'{mod}_spectrum', f'spectra_{mod}']
                    found = False
                    for candidate in candidates:
                        if hasattr(data, candidate):
                            self.spectra_field_names[mod] = candidate
                            found = True
                            break
                    if not found:
                        print(f"Warning: could not find the {mod} spectrum field.")
                        self.spectra_field_names[mod] = mod
        
        spectra_list = []
        for mod in self.modalities:
            field_name = self.spectra_field_names.get(mod, mod)
            if hasattr(data, field_name):
                spectrum = getattr(data, field_name)
                if spectrum.dim() > 1:
                    spectrum = spectrum.squeeze(0)
                    if spectrum.dim() > 1:
                        spectrum = spectrum.flatten()
                spectra_list.append(spectrum)
            else:
                print(f"Warning: sample {idx} is missing the {mod} spectrum. Filling zeros.")
                default_lengths = {'uv': 701, 'ir': 3501, 'raman': 3501}
                default_len = default_lengths.get(mod, 256)
                spectra_list.append(torch.zeros(default_len))
        
        data.spectra = spectra_list
        
        return data

    def download(self):
        """Download is not needed for the local processed release."""
        pass

    def process(self):
        """Processing is not needed for the local processed release."""
        pass
    
    def get_idx_split(self, split_file=None):
        """
        Load or generate a train/valid/test split.
        """
        import os.path as osp
        import numpy as np
        
        if split_file is None:
            split_file = 'split_dict_qm9sp.pt'
        
        split_path = osp.join(self.processed_dir, split_file)
        if osp.exists(split_path):
            print(f'Loading existing split data from {split_file}.')
            return torch.load(split_path)
        
        data_num = len(self)
        print(f'Generating new split for {data_num} samples...')
        
        train_num = min(100000, int(0.8 * data_num))
        test_num = int(0.1 * data_num)
        valid_num = data_num - (train_num + test_num)
        
        np.random.seed(0)
        data_perm = np.random.permutation(data_num)
        train, valid, test = np.split(
            data_perm, [train_num, train_num + valid_num])
        
        split_dict = {
            'train': torch.from_numpy(train),
            'valid': torch.from_numpy(valid),
            'test': torch.from_numpy(test)
        }
        
        torch.save(split_dict, split_path)
        print(f'Split saved: train={len(train)}, valid={len(valid)}, test={len(test)}')
        
        return split_dict


class QM9SPTransform:
    """
    Convert raw PyG QM9SP samples into the tensors expected by MAST.
    """
    
    def __init__(self, atom_type_list=None, include_aromatic=False, spectra_normalize=True):
        self.z_to_type = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
        self.type_to_z = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
        self.include_aromatic = include_aromatic
        self.spectra_normalize = spectra_normalize
        
        if atom_type_list is None:
            atom_type_list = [0, 1, 2, 3, 4]
        self.atom_type_list = torch.tensor(list(atom_type_list))
    
    def __call__(self, data):
        """
        Transform a PyG QM9SP sample into the MAST tensor format.
        
        Input (PyG format):
            - z: [N] atomic numbers
            - pos: [N, 3] coordinates
            - edge_index: [2, E]
            - edge_attr: [E, 4] bond type one-hot
            
        Output (MAST format):
            - atom_type: [N] atom type indices
            - atom_one_hot: [N, 5] atom type one-hot
            - pos: [N, 3]
            - edge_one_hot: [N, N, 2] edge features
            - fc: [N] formal charges
            - num_atom: int
            - spectra: list of [L] tensors
        """
        atom_type = torch.tensor([self.z_to_type.get(z.item(), 0) for z in data.z])
        data.atom_type = atom_type
        
        atom_one_hot = atom_type.unsqueeze(-1) == self.atom_type_list.unsqueeze(0)
        data.atom_one_hot = atom_one_hot.float()
        
        N = data.num_nodes
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_type = torch.argmax(data.edge_attr, dim=-1) + 1
            edge_bond = edge_type.float().clone()
            edge_bond[edge_bond == 4] = 0
            edge_bond = edge_bond / 3.
            
            edge_index = data.edge_index
            dense_edge = torch.zeros(N, N, 1)
            dense_edge[edge_index[0], edge_index[1]] = edge_bond.unsqueeze(-1)
            
            edge_exist = (dense_edge.sum(dim=-1, keepdim=True) != 0).float()
            dense_edge_one_hot = torch.cat([edge_exist, dense_edge], dim=-1)
        else:
            print("Warning: edge_attr is missing.")
            dense_edge_one_hot = torch.zeros(N, N, 2)
        
        data.edge_one_hot = dense_edge_one_hot
        
        data.fc = torch.zeros(N)
        
        data.num_atom = N
        
        if hasattr(data, 'spectra') and data.spectra is not None:
            if self.spectra_normalize:
                normalized_spectra = []
                for spectrum in data.spectra:
                    mean = spectrum.mean()
                    std = spectrum.std()
                    if std > 1e-8:
                        spectrum = (spectrum - mean) / std
                    normalized_spectra.append(spectrum)
                data.spectra = normalized_spectra
        
        data.rdmol = self._create_rdmol(data)
        
        return data
    
    def _create_rdmol(self, data):
        """
        Build an RDKit molecule for evaluation utilities.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            import copy
            
            mol = Chem.RWMol()
            
            for z_val in data.z:
                atom_num = int(z_val.item())
                atom = Chem.Atom(atom_num)
                mol.AddAtom(atom)
            
            if hasattr(data, 'edge_index') and hasattr(data, 'edge_attr'):
                edge_index = data.edge_index
                edge_attr = data.edge_attr
                
                added_bonds = set()
                
                for i in range(edge_index.shape[1]):
                    src = int(edge_index[0, i].item())
                    dst = int(edge_index[1, i].item())
                    
                    if src >= dst:
                        continue
                    
                    bond_key = (min(src, dst), max(src, dst))
                    if bond_key in added_bonds:
                        continue
                    added_bonds.add(bond_key)
                    
                    bond_type_idx = torch.argmax(edge_attr[i]).item()
                    bond_types = [
                        Chem.BondType.SINGLE,
                        Chem.BondType.DOUBLE,
                        Chem.BondType.TRIPLE,
                        Chem.BondType.AROMATIC
                    ]
                    bond_type = bond_types[bond_type_idx]
                    
                    mol.AddBond(src, dst, bond_type)
            
            if hasattr(data, 'pos'):
                conf = Chem.Conformer(data.num_nodes)
                for i, pos in enumerate(data.pos):
                    x, y, z = pos.tolist()
                    conf.SetAtomPosition(i, (float(x), float(y), float(z)))
                mol.AddConformer(conf)
            
            mol = mol.GetMol()
            
            try:
                Chem.SanitizeMol(mol)
            except:
                pass
            
            return copy.deepcopy(mol)
            
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to build rdmol: {e}")
            return None


def collate_qm9sp(items):
    """
    Collate QM9SP `Data` objects into padded batch tensors.
    """
    batch_data = []
    for item in items:
        batch_data.append((
            item.atom_one_hot,
            item.edge_one_hot,
            item.fc,
            item.pos,
            item.num_atom,
            item.spectra
        ))
    
    atom_one_hot, edge_one_hot, formal_charges, positions, num_atoms, spectra_list = zip(*batch_data)
    
    max_node_num = max(num_atoms)
    
    def pad_node_feature(x, pad_len):
        x_len, x_dim = x.size()
        if x_len < pad_len:
            new_x = x.new_zeros([pad_len, x_dim], dtype=x.dtype)
            new_x[:x_len, :] = x
            x = new_x
        return x.unsqueeze(0)
    
    def pad_edge_feature(x, pad_len):
        x_len, _, x_dim = x.size()
        if x_len < pad_len:
            new_x = x.new_zeros([pad_len, pad_len, x_dim])
            new_x[:x_len, :x_len, :] = x
            x = new_x
        return x.unsqueeze(0)
    
    def get_node_mask(node_num, pad_len, dtype):
        node_mask = torch.zeros(pad_len, dtype=dtype)
        node_mask[:node_num] = 1.
        return node_mask.unsqueeze(0)
    
    atom_one_hot = torch.cat([pad_node_feature(i, max_node_num) for i in atom_one_hot])
    formal_charges = torch.cat([pad_node_feature(i.unsqueeze(-1), max_node_num) for i in formal_charges])
    positions = torch.cat([pad_node_feature(i, max_node_num) for i in positions])
    edge_one_hot = torch.cat([pad_edge_feature(i, max_node_num) for i in edge_one_hot])
    
    if spectra_list[0] is not None and len(spectra_list[0]) > 0:
        num_modalities = len(spectra_list[0])
        batch_spectra = []
        for mod_idx in range(num_modalities):
            mod_batch = torch.stack([spec[mod_idx] for spec in spectra_list], dim=0)
            batch_spectra.append(mod_batch)
    else:
        batch_spectra = None
    
    node_mask = torch.cat([get_node_mask(i, max_node_num, atom_one_hot.dtype) for i in num_atoms])
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)
    
    return dict(
        atom_one_hot=atom_one_hot,
        edge_one_hot=edge_one_hot,
        positions=positions,
        formal_charges=formal_charges,
        atom_mask=node_mask,
        edge_mask=edge_mask,
        spectra=batch_spectra
    )
