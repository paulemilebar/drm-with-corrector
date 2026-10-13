"""CPU checks for correction semantics, using tiny networks and no renderer."""

import tempfile
import unittest
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytorch_lightning as pl
import torch
from torch import nn

from models.corrective_drmnet import CorrectiveDRMNet
from models.drmnet import DRMNet


class TinyWrapper(nn.Module):
    def __init__(self, config, conditioning_key, *args):
        super().__init__()
        self.conditioning_key = conditioning_key
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.is_refnet = config == "refnet"
        self.calls = []

    def forward(self, x, embedding, c_concat):
        self.calls.append((x.detach().clone(), embedding.detach().clone()))
        if self.is_refnet:
            return self.weight.expand(x.shape[0], 6)
        return self.weight * x + 0.1 * c_concat[0]


class Identity(nn.Module):
    def encode(self, x):
        return x


def no_renderer(model, config):
    model.register_buffer("basis_r0", torch.ones(3, 2, 2), persistent=False)


class CorrectiveDRMNetTests(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("models.drmnet.ZEmbDiffusionWrapper", TinyWrapper),
            ("models.drmnet.DiffusionWrapper", TinyWrapper),
            ("models.drmnet.instantiate_from_config", lambda config: Identity()),
            ("models.drmnet.DRMNet.instantiate_brdf_model", no_renderer),
            ("models.corrective_drmnet.CorrectiveDRMNet.instantiate_brdf_model", no_renderer),
        ):
            patcher = patch(name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.kwargs = dict(illnet_config="illnet", refnet_config="refnet", renderer_config={},
                           concat_mode=True, loss_type="l2", sigma=0.0, gamma=0.5, epsilon=0.1,
                           max_timesteps=12, z0=[1, 1, 1, 1, 0, 1])
        base = DRMNet(**self.kwargs)
        # Deliberately different live and EMA weights to verify initialization.
        with torch.no_grad():
            base.illnet_model.weight.fill_(0.8)
        self.checkpoint = Path(self.tmp.name) / "base.ckpt"
        torch.save({"state_dict": base.state_dict()}, self.checkpoint)
        self.model = CorrectiveDRMNet(ckpt_path=self.checkpoint, **self.kwargs)
        self.model.learning_rate = 0.01
        self.x = torch.ones(2, 3, 2, 2)
        self.cond = [torch.full_like(self.x, 0.5)]
        self.K, self.k = torch.tensor([4, 4]), torch.tensor([2, 2])

    def loss(self, target=None):
        return self.model.p_losses(self.x, self.x if target is None else target,
                                   None, None, self.K, self.k, self.cond, self.cond)[0]

    def test_pretrained_ema_freeze_and_optimizer(self):
        model = self.model
        torch.testing.assert_close(model.illnet_model.weight, torch.tensor(0.2))
        torch.testing.assert_close(model.corrector.model.weight, model.illnet_model.weight)
        self.assertNotEqual(model.corrector.model.weight.data_ptr(), model.illnet_model.weight.data_ptr())
        model.eval().train()
        self.assertFalse(model.illnet_model.training)
        self.assertFalse(model.refnet_model.training)
        self.assertTrue(model.corrector.training)
        self.assertEqual({id(p) for p in model.parameters() if p.requires_grad},
                         {id(p) for group in model.configure_optimizers().param_groups for p in group["params"]})
        raw = CorrectiveDRMNet(ckpt_path=self.checkpoint, base_use_ema=False, **self.kwargs)
        torch.testing.assert_close(raw.illnet_model.weight, torch.tensor(0.8))

    def test_zero_gate_preserves_entire_vanilla_sampler(self):
        model = self.model.eval()
        model.corrections_per_step = 0
        torch.manual_seed(17)
        expected = model.p_sample_loop(self.x, self.cond, self.cond, verbose=False)
        model.corrections_per_step = 2
        torch.manual_seed(17)
        actual = model.p_sample_loop(self.x, self.cond, self.cond, verbose=False)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        self.assertTrue(torch.isfinite(actual[1]).all())

    def test_same_step_blend_and_one_refnet_call(self):
        model = self.model.eval()
        model.corrections_per_step = 2
        with torch.no_grad():
            model.corrector.gate.fill_(0.5)
            model.corrector.model.weight.zero_()
        mean, _, _, residual = model.p_mean_variance(self.x, self.cond, self.cond, 1, True)
        # Base q=.25, corrector q=.05: q1=.15, q2=.10, always relative to x.
        torch.testing.assert_close(residual, torch.full_like(self.x, 0.1))
        torch.testing.assert_close(mean, self.x + 0.1)
        calls = model.corrector.model.calls
        self.assertEqual(len(model.refnet_model.calls), 1)
        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0][0], self.x + 0.25)
        torch.testing.assert_close(calls[1][0], self.x + 0.15)
        torch.testing.assert_close(calls[0][1], calls[1][1])

    def test_only_corrector_learns_and_training_uses_one_pass(self):
        model = self.model
        model.corrections_per_step = 2
        optimizer = model.configure_optimizers()
        initial_corrector = model.corrector.model.weight.detach().clone()
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        for _ in range(2):
            model.corrector.model.calls.clear()
            optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            self.assertEqual(len(model.corrector.model.calls), 1)
            self.assertIsNotNone(model.corrector.gate.grad)
            self.assertIsNotNone(model.corrector.model.weight.grad)
            for name, param in model.named_parameters():
                if name in frozen:
                    self.assertIsNone(param.grad)
                    torch.testing.assert_close(param, frozen[name])
            optimizer.step()
            model.on_train_batch_end()
        self.assertFalse(torch.equal(model.corrector.model.weight, initial_corrector))
        self.assertNotEqual(model.corrector.gate.item(), 0.0)

    def test_corrective_checkpoint_and_ema_restore(self):
        model = self.model
        with torch.no_grad():
            model.corrector.gate.fill_(0.4)
        model.on_train_batch_end()
        live = model.corrector.gate.detach().clone()
        with model.ema_scope():
            self.assertNotEqual(model.corrector.gate.item(), live.item())
        torch.testing.assert_close(model.corrector.gate, live)
        path = Path(self.tmp.name) / "corrector.ckpt"
        torch.save({"state_dict": model.state_dict()}, path)
        restored = CorrectiveDRMNet(ckpt_path=path, **self.kwargs)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)

    def test_no_transition_and_no_gt_material_conditioning(self):
        self.K.zero_()
        loss = self.loss()
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        with self.assertRaises(ValueError):
            CorrectiveDRMNet(train_with_zk_gt=True, **self.kwargs)

    def test_lightning_fit_and_resume_optimizer(self):
        class Data(pl.LightningDataModule):
            def __init__(self):
                super().__init__()
                self.train_ds = torch.utils.data.TensorDataset(torch.zeros(4))
                self.val_ds = self.train_ds

            def train_dataloader(self):
                return torch.utils.data.DataLoader(self.train_ds, batch_size=2)

            def val_dataloader(self):
                return torch.utils.data.DataLoader(self.val_ds, batch_size=2)

        def get_input(model, batch):
            model.batch_size = len(self.x)
            return self.K, self.k, self.x, self.x, self.x, None, None, self.cond, self.cond

        self.model.get_input = MethodType(get_input, self.model)
        options = dict(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                       limit_val_batches=1)
        trainer = pl.Trainer(max_steps=2, **options)
        trainer.fit(self.model, datamodule=Data())
        self.assertIn("val/loss_ema", trainer.callback_metrics)
        checkpoint = Path(self.tmp.name) / "resume.ckpt"
        trainer.save_checkpoint(checkpoint)
        restored = CorrectiveDRMNet(ckpt_path=self.checkpoint, **self.kwargs)
        restored.learning_rate = self.model.learning_rate
        restored.get_input = MethodType(get_input, restored)
        resumed = pl.Trainer(max_steps=3, **options)
        resumed.fit(restored, datamodule=Data(), ckpt_path=checkpoint)
        self.assertEqual(resumed.global_step, 3)
        for state in resumed.optimizers[0].state.values():
            self.assertEqual(state["step"].item(), 3)
        torch.testing.assert_close(restored.illnet_model.weight, self.model.illnet_model.weight)


if __name__ == "__main__":
    unittest.main()
