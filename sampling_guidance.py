import re

import torch
import torch.nn.functional as F
from torch_scatter import scatter


class MolSpectraGuidance:
    """Reward and guidance encoder used during MCTS inference."""

    def __init__(self, ckpt_path, device="cuda"):
        self.device = device
        self.model = None
        self.hparams = None
        self._load_model(ckpt_path)

    def _load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.hparams = ckpt["hyper_parameters"]
        state_dict = ckpt["state_dict"]

        from torchmdnet.models.model import create_model

        self.hparams["spectra_model"] = "SpecFormer"
        self.hparams["output_model_spec"] = None
        self.hparams["output_model_mol"] = None
        self.hparams.setdefault("use_dataset_md17", False)
        self.hparams.setdefault("layernorm_on_vec", None)
        self.hparams.setdefault("derivative", False)
        self.hparams.setdefault("prior_model", None)
        self.hparams.setdefault("output_model_noise", "VectorOutput")
        self.hparams.setdefault("position_noise_scale", 0.0)

        self.model = create_model(self.hparams)
        new_state_dict = {re.sub(r"^model\.", "", k): v for k, v in state_dict.items()}

        current_model_dict = self.model.state_dict()
        filtered_state_dict = {}
        for key in current_model_dict:
            if key in new_state_dict and current_model_dict[key].size() == new_state_dict[key].size():
                filtered_state_dict[key] = new_state_dict[key]
            else:
                filtered_state_dict[key] = current_model_dict[key]

        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode_spectra(self, spectra_list):
        spectra_list = [s.to(self.device) for s in spectra_list]
        spec_feature = self.model.representation_spec_model(spectra_list)
        if isinstance(spec_feature, tuple):
            spec_feature = spec_feature[0]
        return spec_feature

    def encode_structure(self, z, pos, batch):
        z = z.to(self.device)
        batch = batch.to(self.device)
        if pos.device != torch.device(self.device):
            pos = pos.to(self.device)
        if not pos.requires_grad:
            pos = pos.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            x, _, _, _, batch_out = self.model.representation_model(z, pos, batch=batch)
            mol_feature = scatter(x, batch_out, dim=0, reduce=self.model.reduce_op)
        return mol_feature

    def compute_guidance_loss(self, spec_feature, mol_feature):
        spec_norm = F.normalize(spec_feature, p=2, dim=-1)
        mol_norm = F.normalize(mol_feature, p=2, dim=-1)
        cosine_sim = (spec_norm * mol_norm).sum(dim=-1)
        return -cosine_sim.mean()
