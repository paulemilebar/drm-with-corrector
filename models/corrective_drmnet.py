"""A frozen vanilla DRMNet with prediction correction (paper Eq. 7).

Correction refines the same k -> k-1 residual; it never advances the sampler.
Training uses the vanilla synthetic states, not on-policy trajectory rollouts.
"""

from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from ldm.modules.ema import LitEma
from ldm.util import instantiate_from_config
from models.drmnet import DRMNet


class _Corrector(nn.Module):
    def __init__(self, illnet, gate_init):
        super().__init__()
        self.model = deepcopy(illnet).requires_grad_(True)
        # Include the gate in the corrector's optimizer, checkpoint and EMA.
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))


class CorrectiveDRMNet(DRMNet):
    def __init__(
        self,
        ckpt_path=None,
        base_use_ema=True,
        use_ema=True,
        corrector_enabled=True,
        corrector_gate_init=0.0,
        corrector_loss_weight=None,
        train_corrections_per_step=1,
        corrections_per_step=1,
        **kwargs,
    ):
        if type(train_corrections_per_step) is not int or train_corrections_per_step < 1:
            raise ValueError("train_corrections_per_step must be a positive integer")
        if type(corrections_per_step) is not int or corrections_per_step < 0:
            raise ValueError("corrections_per_step must be a nonnegative integer")
        if kwargs.get("train_with_zk_gt", False) or kwargs.get("train_with_zk_gt_switch_epoch") is not None:
            raise ValueError("Correction uses RefNet predictions: disable train_with_zk_gt and its switch epoch")
        if kwargs.get("cond_stage_trainable", False) or kwargs.get("ignore_keys"):
            raise ValueError("Correction requires a frozen conditioner and a complete vanilla checkpoint")

        # Base EMA must be loaded/copied before freezing: LitEma.copy_to checks
        # requires_grad. A corrective checkpoint is restored after C exists.
        super().__init__(ckpt_path=None, use_ema=base_use_ema, **kwargs)
        state = None
        corrective_checkpoint = False
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            state = checkpoint.get("state_dict", checkpoint)
            corrective_checkpoint = any(key.startswith("corrector.") for key in state)
            if not corrective_checkpoint:
                # When explicitly choosing raw base weights, omit unused EMAs.
                base_state = state if base_use_ema else {
                    key: value for key, value in state.items()
                    if not key.startswith(("illnet_model_ema.", "refnet_model_ema."))
                }
                self.load_state_dict(base_state, strict=True)
        if base_use_ema:
            self.illnet_model_ema.copy_to(self.illnet_model)
            self.refnet_model_ema.copy_to(self.refnet_model)
            del self.illnet_model_ema, self.refnet_model_ema

        self.requires_grad_(False)
        self.corrector = _Corrector(self.illnet_model, corrector_gate_init)
        self.use_ema = use_ema
        if use_ema:
            self.corrector_ema = LitEma(self.corrector)
        self.corrector_enabled = corrector_enabled
        self.train_corrections_per_step = train_corrections_per_step
        self.corrections_per_step = corrections_per_step
        self.corrector_loss_weight = self.l_refmap_weight if corrector_loss_weight is None else corrector_loss_weight
        if corrective_checkpoint:
            self.load_state_dict(state, strict=True)
        if ckpt_path is not None:
            print(f"Loaded {'corrective' if corrective_checkpoint else 'vanilla'} checkpoint: {ckpt_path}")
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.illnet_model.eval()
        self.refnet_model.eval()
        return self

    def get_brdf_out(self, brdf_model_out, reversed_k=None):
        # Use the vanilla inference policy even while the corrector is training.
        _, _, zk = self.get_schedule(brdf_model_out, reversed_k=reversed_k)
        return zk.clamp(0, 1), brdf_model_out.clamp(0, 1)

    def _corrected_residual(self, x, illnet_cond, refnet_cond, reversed_k, steps):
        with torch.no_grad():
            residual, z_out = super().forward(x, illnet_cond, refnet_cond, reversed_k)
            zk, _ = self.get_brdf_out(z_out, reversed_k)
            embedding = zk - self.z0
        for _ in range(steps):
            # Keep the current-step embedding and x fixed for every correction.
            # Full stop-gradient applies to both paths of the paper's blend.
            proposal = x + residual
            prediction = self.apply_model(self.corrector.model, proposal.detach(), embedding, illnet_cond)
            residual = (1 - self.corrector.gate) * residual.detach() + self.corrector.gate * prediction
        return residual, z_out

    def p_losses(self, Lr_k, Lr_km1, z_k, z_K, K, k, illnet_cond, refnet_cond):
        if self.sigma > 0:
            Lr_k = Lr_k + self.sigma * torch.randn_like(Lr_k)
        # Supervise the final correction; the sampling count is independent.
        residual, _ = self._corrected_residual(
            Lr_k, illnet_cond, refnet_cond, K - k - 1, steps=self.train_corrections_per_step
        )
        valid = K != 0
        if valid.any():
            loss_refmap = self.get_loss(residual[valid], (Lr_km1 - Lr_k)[valid])
        else:
            # Keep a differentiable zero when a batch contains no transition.
            loss_refmap = residual.sum() * 0
        loss = self.corrector_loss_weight * loss_refmap
        prefix = "train" if self.training else "val"
        return loss, {
            f"{prefix}/loss": loss,
            f"{prefix}/loss_refmap": loss_refmap,
            f"{prefix}/corrector_gate": self.corrector.gate.detach().clone(),
        }

    def p_mean_variance(self, Lr_k, illnet_cond, refnet_cond, reversed_k, return_model_out=False):
        steps = self.corrections_per_step if self.corrector_enabled else 0
        residual, z_out = self._corrected_residual(Lr_k, illnet_cond, refnet_cond, reversed_k, steps)
        result = (Lr_k + residual, self.delta, z_out)
        return result + (residual,) if return_model_out else result

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.corrector_ema.store(self.corrector.parameters())
            self.corrector_ema.copy_to(self.corrector)
        try:
            yield
        finally:
            if self.use_ema:
                self.corrector_ema.restore(self.corrector.parameters())

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.corrector_ema(self.corrector)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.corrector.parameters(), lr=self.learning_rate)
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)
            return [optimizer], [{"scheduler": LambdaLR(optimizer, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
        return optimizer

    def instantiate_brdf_model(self, config):
        # Same basis render as vanilla; instance transforms support Mitsuba 3.6
        # in the supplied SIF without changing models/drmnet.py.
        import mitsuba as mi
        from utils.mitsuba3_utils import scalar_look_at

        self.renderer = instantiate_from_config(config)
        sensor = mi.load_dict({
            "type": "refmapsensor",
            "to_world": scalar_look_at([0, 0, 1.1]),
            "film": {"type": "hdrfilm", "width": self.image_size, "height": self.image_size, "rfilter": {"type": "box"}},
            "sampler": {"type": "stratified", "sample_count": self.renderer.spp},
        })
        basis = self.renderer.rendering(self.z0, self.brdf_param_names,
                                        envmap=torch.ones(*self.renderer.envmap_size, 3), sensor=sensor)
        self.register_buffer("basis_r0", basis.permute(2, 0, 1).contiguous(), persistent=False)
