from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import torch.nn as nn


# Baseline defaults before the current aggregator sweep:
#   sigmas = (1.5, 3.0, 6.0, 12.0)
#   ell_centroids = (20.0, 15.0, 10.0, 6.0, 3.0)
#   freq_c = 3.5

# Valeurs for the current EnvNet trained UViT_baseline_expressif:
 # sigmas: tuple[float, ...] = (1.5, 3.0, 6.0, 15.0),
 #     ell_centroids: Optional[Sequence[float]] = (30.0, 20.0, 14.0, 8.0, 4.0),
 #       freq_alpha: float = 1.75,
 #       freq_c: float = 2.75,


class ConsensusAggregatorMultiplicatif(nn.Module):
    """Multiplicative credentialed consensus aggregator.

    For M=1, the consensus is simply the single agent illumination estimate
    from the previous denoising step, preserving DRM equivalence.

    For M>1, the aggregator builds the same spherical Gaussian scale-space as
    the additive Laplacian aggregator and represents each historical band by
    a log ratio between the exact same adjacent blur levels. The frequency and spatial
    credentials are intentionally identical to :class:`ConsensusAggregator`;
    only the frequency representation and coarse-to-fine reconstruction differ.
    Returns:
        consensus_env: (B, 3, Henv, Wenv)  shared illumination estimate on S²_ω
    """

    def __init__(
        self,
        sigmas: tuple[float, ...] = (1.5, 3.0, 6.0, 15.0),
        roughness_index: int = 4,
        frequency_credential_mode: str = "all_parameters",
        ell_centroids: Optional[Sequence[float]] = (30.0, 20.0, 14.0, 8.0, 4.0),
        freq_alpha: float = 1.75,
        freq_c: float = 2.75,
        ell_min: float = 1.0,
        ell_cap: float = 64.0,
        metallic_index: int = 0,
        specular_index: int = 5,
        use_midfreq_material_boost: bool = True,
        midfreq_boost_weight: float = 5.6,
        midfreq_specular_scale: float = 0.5,
        midfreq_roughness_power: float = 2.0,
        midfreq_material_power: float = 4.0,
        midfreq_material_threshold: float = 0.10,
        midfreq_profile: Optional[Sequence[float]] = None,
        freq_weight_cap: Optional[float] = None,
        frequency_credential_beta: float = 1.0,
        use_midfreq_highfreq_coherence: bool = False,
        midfreq_highfreq_coherence_weight: float = 1.0,
        midfreq_highfreq_profile: Optional[Sequence[float]] = None,
        use_band_energy_normalization: bool = True,
        sigma_min_deg: float = 3.0,
        sigma_max_deg: float = 30.0,
        normal_chunk_size: int = 256,
        spatial_truncation_sigma: float = 2.0,
        max_phi_window: Optional[int] = None,
        support_threshold: float = 1.0e-6,
        eps: float = 1.0e-6,
        multiplicative_eps: float = 1.0e-6,
        max_log_ratio: float = 10.0,
        use_energy_anchor: bool = False,
        energy_scale_min: Optional[float] = None,
        energy_scale_max: Optional[float] = None,
    ) -> None:
        super().__init__()
        if len(sigmas) == 0:
            raise ValueError("ConsensusAggregatorMultiplicatif requires at least one Gaussian sigma")
        if normal_chunk_size <= 0:
            raise ValueError("normal_chunk_size must be positive")
        if spatial_truncation_sigma <= 0:
            raise ValueError("spatial_truncation_sigma must be positive")
        if max_phi_window is not None and max_phi_window <= 0:
            raise ValueError("max_phi_window must be positive when provided")
        valid_frequency_modes = {"roughness_only", "all_parameters"}
        if frequency_credential_mode not in valid_frequency_modes:
            raise ValueError(
                "frequency_credential_mode must be one of "
                f"{sorted(valid_frequency_modes)}, got {frequency_credential_mode!r}"
            )
        if freq_weight_cap is not None and freq_weight_cap <= 0:
            raise ValueError("freq_weight_cap must be positive when provided")
        if not 0.0 < frequency_credential_beta <= 1.0:
            raise ValueError("frequency_credential_beta must be in (0, 1]")
        if midfreq_material_power <= 0:
            raise ValueError("midfreq_material_power must be positive")
        if not 0.0 <= midfreq_material_threshold < 1.0:
            raise ValueError("midfreq_material_threshold must be in [0, 1)")
        if multiplicative_eps <= 0:
            raise ValueError("multiplicative_eps must be positive")
        if max_log_ratio <= 0:
            raise ValueError("max_log_ratio must be positive")
        if energy_scale_min is not None and energy_scale_min <= 0:
            raise ValueError("energy_scale_min must be positive when provided")
        if energy_scale_max is not None and energy_scale_max <= 0:
            raise ValueError("energy_scale_max must be positive when provided")
        if energy_scale_min is not None and energy_scale_max is not None and energy_scale_min > energy_scale_max:
            raise ValueError("energy_scale_min cannot exceed energy_scale_max")
        self.sigmas = tuple(float(s) for s in sigmas)
        num_bands = len(self.sigmas) + 1
        if ell_centroids is None:
            if num_bands == 5:
                ell_centroids = (20.0, 15.0, 10.0, 6.0, 3.0)
            else:
                ell_centroids = torch.logspace(
                    math.log10(32.0),
                    math.log10(1.0),
                    steps=num_bands,
                ).tolist()
        if len(ell_centroids) != num_bands:
            raise ValueError(
                f"ell_centroids length {len(ell_centroids)} must match "
                f"number of Laplacian bands {num_bands}"
            )
        self.ell_centroids = tuple(float(ell) for ell in ell_centroids)
        self.roughness_index = int(roughness_index)
        self.frequency_credential_mode = str(frequency_credential_mode)
        self.metallic_index = int(metallic_index)
        self.specular_index = int(specular_index)
        self.freq_alpha = float(freq_alpha)
        self.freq_c = float(freq_c)
        self.ell_min = float(ell_min)
        self.ell_cap = float(ell_cap)
        self.use_midfreq_material_boost = bool(use_midfreq_material_boost)
        self.midfreq_boost_weight = float(midfreq_boost_weight)
        self.midfreq_specular_scale = float(midfreq_specular_scale)
        self.midfreq_roughness_power = float(midfreq_roughness_power)
        self.midfreq_material_power = float(midfreq_material_power)
        self.midfreq_material_threshold = float(midfreq_material_threshold)
        if midfreq_profile is None and num_bands == 5:
            midfreq_profile = (0.14, 0.20, 0.40, 0.88, 0.0)
        if midfreq_profile is not None and len(midfreq_profile) != num_bands:
            raise ValueError(
                f"midfreq_profile length {len(midfreq_profile)} must match "
                f"number of Laplacian bands {num_bands}"
            )
        self.midfreq_profile = (
            None if midfreq_profile is None else tuple(float(v) for v in midfreq_profile)
        )
        self.freq_weight_cap = None if freq_weight_cap is None else float(freq_weight_cap)
        self.frequency_credential_beta = float(frequency_credential_beta)
        self.use_midfreq_highfreq_coherence = bool(use_midfreq_highfreq_coherence)
        self.midfreq_highfreq_coherence_weight = float(midfreq_highfreq_coherence_weight)
        if midfreq_highfreq_profile is None and num_bands == 5:
            midfreq_highfreq_profile = (0.45, 0.30, 0.0, 0.0, 0.0)
        if midfreq_highfreq_profile is not None and len(midfreq_highfreq_profile) != num_bands:
            raise ValueError(
                f"midfreq_highfreq_profile length {len(midfreq_highfreq_profile)} must match "
                f"number of Laplacian bands {num_bands}"
            )
        self.midfreq_highfreq_profile = (
            None
            if midfreq_highfreq_profile is None
            else tuple(float(v) for v in midfreq_highfreq_profile)
        )
        self.use_band_energy_normalization = bool(use_band_energy_normalization)
        self.sigma_min_deg = float(sigma_min_deg)
        self.sigma_max_deg = float(sigma_max_deg)
        self.normal_chunk_size = int(normal_chunk_size)
        self.spatial_truncation_sigma = float(spatial_truncation_sigma)
        self.max_phi_window = None if max_phi_window is None else int(max_phi_window)
        self.support_threshold = float(support_threshold)
        self.eps = float(eps)
        self.multiplicative_eps = float(multiplicative_eps)
        self.max_log_ratio = float(max_log_ratio)
        self.use_energy_anchor = bool(use_energy_anchor)
        self.energy_scale_min = energy_scale_min
        self.energy_scale_max = energy_scale_max
        self.last_debug: dict[str, object] = {}
        self._last_spatial_debug: dict[str, object] = {}
        self._last_frequency_debug: dict[str, object] = {}
        self._collect_frequency_debug = False

    def forward(
        self,
        envmaps: torch.Tensor,
        z: torch.Tensor,
        normal_map: Optional[torch.Tensor] = None,
        normal_mask: Optional[torch.Tensor] = None,
        _collect_debug: bool = False,
        _return_normalized_weights: bool = False,
        credential_cache: Optional[dict[str, object]] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(envmaps, z, normal_map, normal_mask)
        if envmaps.shape[1] == 1:
            self.last_debug = {}
            self._last_frequency_debug = {}
            if _return_normalized_weights:
                alpha = envmaps.new_ones(
                    envmaps.shape[0], 1, len(self.sigmas) + 1, 1,
                    envmaps.shape[-2], envmaps.shape[-1],
                )
                return envmaps[:, 0], alpha
            return envmaps[:, 0]

        output_dtype = envmaps.dtype
        compute_dtype = torch.float32 if envmaps.dtype in (torch.float16, torch.bfloat16) else envmaps.dtype
        envmaps_work = envmaps.to(dtype=compute_dtype)
        z_work = z.to(dtype=compute_dtype)
        normal_map_work = None if normal_map is None else normal_map.to(dtype=compute_dtype)
        normal_mask_work = None if normal_mask is None else normal_mask.to(dtype=compute_dtype)
        pyramid, log_ratios = self.multiplicative_pyramid(envmaps_work)
        num_bands = len(log_ratios) + 1

        if credential_cache is None:
            self._collect_frequency_debug = bool(_collect_debug)
            w_freq_raw, ell_max, ell_centroids = self.frequency_credentials(z_work, num_bands, compute_dtype)
            w_freq = self._apply_frequency_credential_beta(w_freq_raw)
            frequency_debug = dict(self._last_frequency_debug)
            # Spatial credentials are deterministic geometric weights with no
            # trainable parameters.  Building their geodesic kernels under
            # autograd retains every normal chunk until backward and can consume
            # tens of GiB.  Keep the exact forward values while treating them as
            # fixed weights; gradients through envmaps and w_freq remain intact.
            with torch.no_grad():
                w_spat = self.spatial_credentials(
                    envmaps_work.detach(),
                    z_work.detach(),
                    None if normal_map_work is None else normal_map_work.detach(),
                    None if normal_mask_work is None else normal_mask_work.detach(),
                )
            spatial_debug = dict(self._last_spatial_debug)
            if not _collect_debug:
                self._last_spatial_debug = {}
                self._last_frequency_debug = {}
            weights = w_freq[:, :, :, None, None, None] * w_spat[:, :, None]
            denominator = weights.sum(dim=1)
        else:
            weights = credential_cache["weights"].to(device=envmaps.device, dtype=compute_dtype)
            denominator = credential_cache["denominator"].to(device=envmaps.device, dtype=compute_dtype)
            w_freq = credential_cache["w_freq"].to(device=envmaps.device, dtype=compute_dtype)
            w_freq_raw = credential_cache.get("w_freq_raw", w_freq).to(
                device=envmaps.device,
                dtype=compute_dtype,
            )
            w_spat = credential_cache["w_spat"].to(device=envmaps.device, dtype=compute_dtype)
            ell_centroids = credential_cache["ell_centroids"].to(device=envmaps.device, dtype=compute_dtype)
            ell_max = credential_cache["ell_max"].to(device=envmaps.device, dtype=compute_dtype)
            spatial_debug = credential_cache.get("spatial_debug", {})
            frequency_debug = {}
            self._check_credential_cache(weights, denominator, envmaps_work, num_bands)

        alpha = self._normalize_weights(weights)
        multiscale_weights: list[torch.Tensor] = []
        fused_log_ratios: list[torch.Tensor] = []
        for band_index, ratio in enumerate(log_ratios):
            alpha_level = self._resize_and_normalize_weights(weights[:, :, band_index], ratio.shape[-2:])
            multiscale_weights.append(alpha_level)
            fused_log_ratios.append((alpha_level * ratio).sum(dim=1))

        coarse = pyramid[-1]
        alpha_coarse = self._resize_and_normalize_weights(weights[:, :, -1], coarse.shape[-2:])
        multiscale_weights.append(alpha_coarse)
        fused_coarse = (alpha_coarse * coarse).sum(dim=1)

        consensus = fused_coarse
        for ratio in reversed(fused_log_ratios):
            consensus = self._upsample(consensus, ratio.shape[-2:])
            consensus = consensus * torch.exp(ratio.clamp(-self.max_log_ratio, self.max_log_ratio))
        consensus = consensus.clamp_min(self.multiplicative_eps)

        energy_before_anchor = self._spherical_energy(consensus)
        target_energy = self._agent_energy_anchor(envmaps_work, alpha_coarse)
        anchor_scale = torch.ones_like(energy_before_anchor)
        if self.use_energy_anchor:
            anchor_scale = target_energy / energy_before_anchor.clamp_min(self.multiplicative_eps)
            if self.energy_scale_min is not None:
                anchor_scale = anchor_scale.clamp_min(self.energy_scale_min)
            if self.energy_scale_max is not None:
                anchor_scale = anchor_scale.clamp_max(self.energy_scale_max)
            consensus = consensus * anchor_scale[:, :, None, None]
        energy_after_anchor = self._spherical_energy(consensus)

        reconstruction_errors = []
        for agent_index in range(envmaps.shape[1]):
            reconstructed = pyramid[-1][:, agent_index]
            for ratio in reversed(log_ratios):
                reconstructed = self._upsample(reconstructed, ratio.shape[-2:])
                reconstructed = reconstructed * torch.exp(ratio[:, agent_index])
            reconstruction_errors.append(
                (reconstructed - (envmaps_work[:, agent_index].clamp_min(0) + self.multiplicative_eps)).abs().amax()
            )
        reconstruction_error = torch.stack(reconstruction_errors).amax()
        log_ratio_max_abs = torch.stack([ratio.abs().amax() for ratio in log_ratios])
        log_ratio_clip_fraction = torch.stack([
            (ratio.abs() > self.max_log_ratio).to(dtype=compute_dtype).mean()
            for ratio in log_ratios
        ])

        if _collect_debug:
            self.last_debug = {
                "pyramid": pyramid,
                "log_ratio_bands": log_ratios,
                "coarse_level": pyramid[-1],
                "weights": weights,
                "normalized_weights": alpha,
                "multiscale_normalized_weights": multiscale_weights,
                "w_freq": w_freq,
                "w_freq_raw": w_freq_raw,
                "frequency_credential_beta": self.frequency_credential_beta,
                "w_spat": w_spat,
                "denominator": denominator,
                "support_mask": denominator > self.support_threshold,
                "fused_log_ratio_bands": fused_log_ratios,
                "fused_coarse_level": fused_coarse,
                "log_ratio_max_abs": log_ratio_max_abs,
                "log_ratio_clip_fraction": log_ratio_clip_fraction,
                "reconstruction_error": reconstruction_error,
                "ell_centroids": ell_centroids,
                "ell_max": ell_max,
                "energy_before_anchor": energy_before_anchor,
                "target_agent_energy": target_energy,
                "energy_anchor_scale": anchor_scale,
                "energy_after_anchor": energy_after_anchor,
                "use_energy_anchor": self.use_energy_anchor,
                "consensus_min": consensus.amin(),
                "consensus_max": consensus.amax(),
                "consensus_mean": consensus.mean(),
                "consensus_finite_fraction": torch.isfinite(consensus).to(compute_dtype).mean(),
                "consensus": consensus,
                **frequency_debug,
                **spatial_debug,
            }
        else:
            self.last_debug = {}
        consensus = consensus.to(dtype=output_dtype)
        if _return_normalized_weights:
            return consensus, alpha.to(dtype=output_dtype)
        return consensus

    def _validate_inputs(
        self,
        envmaps: torch.Tensor,
        z: torch.Tensor,
        normal_map: Optional[torch.Tensor],
        normal_mask: Optional[torch.Tensor],
    ) -> None:
        if envmaps.ndim != 5 or envmaps.shape[2] != 3:
            raise ValueError(
                "ConsensusAggregatorMultiplicatif expects envmaps (B,M,3,H,W), "
                f"got {tuple(envmaps.shape)}"
            )
        if z.ndim != 3 or z.shape[:2] != envmaps.shape[:2]:
            raise ValueError(
                "ConsensusAggregatorMultiplicatif expects z (B,M,Dpsi) matching envmaps, "
                f"got {tuple(z.shape)} for envmaps {tuple(envmaps.shape)}"
            )
        if normal_map is not None and normal_map.shape[:2] != envmaps.shape[:2]:
            raise ValueError(f"normal_map leading dims must match envmaps, got {tuple(normal_map.shape)}")
        if normal_mask is not None and normal_mask.shape[:2] != envmaps.shape[:2]:
            raise ValueError(f"normal_mask leading dims must match envmaps, got {tuple(normal_mask.shape)}")
        if self.roughness_index >= z.shape[-1]:
            raise ValueError(f"roughness_index={self.roughness_index} is outside z dimension {z.shape[-1]}")

    def _normalize_weights(self, weights: torch.Tensor) -> torch.Tensor:
        denominator = weights.sum(dim=1, keepdim=True)
        # Match ConsensusAggregator exactly. A very small but non-zero support
        # still carries meaningful relative agent credentials and must not be
        # replaced by a uniform distribution. When every agent weight is
        # exactly zero, division by ``tiny`` naturally leaves alpha at zero.
        return weights / denominator.clamp_min(torch.finfo(weights.dtype).tiny)

    def _resize_and_normalize_weights(
        self,
        weights: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        if weights.shape[-2:] != size:
            B, M, C, H, W = weights.shape
            weights = F.interpolate(
                weights.reshape(B * M, C, H, W),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).reshape(B, M, C, *size)
        return self._normalize_weights(weights)

    def forward_debug(
        self,
        envmaps: torch.Tensor,
        z: torch.Tensor,
        normal_map: Optional[torch.Tensor] = None,
        normal_mask: Optional[torch.Tensor] = None,
        credential_cache: Optional[dict[str, object]] = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        consensus = self.forward(
            envmaps,
            z,
            normal_map,
            normal_mask,
            _collect_debug=True,
            credential_cache=credential_cache,
        )
        return consensus, self.last_debug

    @torch.no_grad()
    def build_credential_cache(
        self,
        reference_envmaps: torch.Tensor,
        z: torch.Tensor,
        normal_map: Optional[torch.Tensor],
        normal_mask: Optional[torch.Tensor],
    ) -> dict[str, object]:
        """Precompute z/geometry credentials that are reusable for fixed GT z."""
        if reference_envmaps.ndim != 5 or reference_envmaps.shape[2] != 3:
            raise ValueError(
                "reference_envmaps must be (B,M,3,H,W), "
                f"got {tuple(reference_envmaps.shape)}"
            )
        if z.ndim != 3 or z.shape[:2] != reference_envmaps.shape[:2]:
            raise ValueError(
                f"z must be (B,M,Dpsi) matching reference_envmaps, got {tuple(z.shape)}"
            )
        num_bands = len(self.sigmas) + 1
        self._collect_frequency_debug = False
        w_freq_raw, ell_max, ell_centroids = self.frequency_credentials(
            z,
            num_bands,
            reference_envmaps.dtype,
        )
        w_freq = self._apply_frequency_credential_beta(w_freq_raw)
        w_spat = self.spatial_credentials(reference_envmaps, z, normal_map, normal_mask)
        spatial_debug = dict(self._last_spatial_debug)
        self._last_spatial_debug = {}
        weights = w_freq[:, :, :, None, None, None] * w_spat[:, :, None]
        denominator = weights.sum(dim=1)
        return {
            "weights": weights.detach(),
            "denominator": denominator.detach(),
            "w_freq": w_freq.detach(),
            "w_freq_raw": w_freq_raw.detach(),
            "w_spat": w_spat.detach(),
            "ell_centroids": ell_centroids.detach(),
            "ell_max": ell_max.detach(),
            "spatial_debug": {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in spatial_debug.items()
            },
        }

    def _apply_frequency_credential_beta(self, w_freq: torch.Tensor) -> torch.Tensor:
        """Compress inter-agent frequency ratios before spatial fusion.

        ``beta=1`` deliberately returns the original tensor, preserving the
        historical path exactly. For ``0 < beta < 1``, every positive ratio
        ``f_i / f_j`` becomes ``(f_i / f_j) ** beta`` while agent ordering is
        unchanged.
        """
        if self.frequency_credential_beta == 1.0:
            return w_freq
        return w_freq.clamp_min(torch.finfo(w_freq.dtype).tiny).pow(
            self.frequency_credential_beta
        )

    @staticmethod
    def _check_credential_cache(
        weights: torch.Tensor,
        denominator: torch.Tensor,
        envmaps: torch.Tensor,
        num_bands: int,
    ) -> None:
        expected_weights = (
            envmaps.shape[0],
            envmaps.shape[1],
            num_bands,
            1,
            envmaps.shape[-2],
            envmaps.shape[-1],
        )
        expected_denominator = (
            envmaps.shape[0],
            num_bands,
            1,
            envmaps.shape[-2],
            envmaps.shape[-1],
        )
        if weights.shape != expected_weights:
            raise ValueError(f"credential weights must be {expected_weights}, got {tuple(weights.shape)}")
        if denominator.shape != expected_denominator:
            raise ValueError(
                f"credential denominator must be {expected_denominator}, got {tuple(denominator.shape)}"
            )

    def multiplicative_pyramid(
        self,
        envmaps: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return the Gaussian scale-space as positive log ratios.

        The additive bands are ``X-G(s0)``, ``G(s0)-G(s1)``, ..., ``G(sN)``.
        Here the exact same boundaries become ``log(X/G(s0))``,
        ``log(G(s0)/G(s1))``, ..., with ``G(sN)`` as the positive base.
        """
        positive = envmaps.clamp_min(0.0) + self.multiplicative_eps
        gaussian_levels = [
            self.spherical_gaussian_filter(positive, sigma).clamp_min(self.multiplicative_eps)
            for sigma in self.sigmas
        ]
        levels = [positive, *gaussian_levels]
        log_ratios = [
            torch.log(levels[index]) - torch.log(levels[index + 1])
            for index in range(len(levels) - 1)
        ]
        return levels, log_ratios

    @staticmethod
    def _upsample(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        leading = values.shape[:-3]
        channels = values.shape[-3]
        flat = values.reshape(-1, channels, *values.shape[-2:])
        resized = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
        return resized.reshape(*leading, channels, *size)

    def reconstruct_pyramid(
        self,
        coarse: torch.Tensor,
        log_ratios: Sequence[torch.Tensor],
        clamp_log_ratios: bool = False,
    ) -> torch.Tensor:
        """Reconstruct a positive map from coarse level and fine-to-coarse ratios."""
        output = coarse
        for ratio in reversed(log_ratios):
            output = self._upsample(output, ratio.shape[-2:])
            if clamp_log_ratios:
                ratio = ratio.clamp(-self.max_log_ratio, self.max_log_ratio)
            output = output * torch.exp(ratio)
        return output

    def _spherical_energy(self, values: torch.Tensor) -> torch.Tensor:
        H = values.shape[-2]
        theta = (torch.arange(H, device=values.device, dtype=values.dtype) + 0.5) * (math.pi / H)
        area = torch.sin(theta).view(*([1] * (values.ndim - 2)), H, 1)
        spatial_mean = (values * area).sum(dim=(-2, -1)) / (area.sum() * values.shape[-1])
        return spatial_mean.mean(dim=-1, keepdim=True)

    def _agent_energy_anchor(
        self,
        envmaps: torch.Tensor,
        coarse_weights: torch.Tensor,
    ) -> torch.Tensor:
        B, M = envmaps.shape[:2]
        energies = self._spherical_energy(envmaps.reshape(B * M, 3, *envmaps.shape[-2:])).reshape(B, M)
        agent_weights = coarse_weights.mean(dim=(2, 3, 4))
        agent_weights = agent_weights / agent_weights.sum(dim=1, keepdim=True).clamp_min(self.multiplicative_eps)
        return (agent_weights * energies).sum(dim=1, keepdim=True)

    def frequency_credentials(
        self,
        z: torch.Tensor,
        num_bands: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.frequency_credential_mode == "roughness_only":
            return self.frequency_credentials_roughness_only(z, num_bands, dtype)
        if self.frequency_credential_mode == "all_parameters":
            return self.frequency_credentials_all_parameters(z, num_bands, dtype)
        raise ValueError(f"unknown frequency_credential_mode={self.frequency_credential_mode!r}")

    def frequency_credentials_roughness_only(
        self,
        z: torch.Tensor,
        num_bands: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_bands != len(self.ell_centroids):
            raise ValueError(
                f"ell_centroids length {len(self.ell_centroids)} does not match "
                f"runtime number of bands {num_bands}"
            )
        if self.roughness_index < 0 or self.roughness_index >= z.shape[-1]:
            raise ValueError(
                f"roughness_index={self.roughness_index} is outside z dimension {z.shape[-1]}"
            )
        roughness = z[..., self.roughness_index].to(dtype=dtype).clamp_min(0.0)
        ell_max = (self.freq_c / torch.sqrt(roughness + self.eps)).clamp(
            self.ell_min,
            self.ell_cap,
        )
        ell_centroids = torch.tensor(
            self.ell_centroids,
            device=z.device,
            dtype=dtype,
        )
        w_freq = torch.sigmoid(self.freq_alpha * (ell_max[..., None] - ell_centroids))
        if self.freq_weight_cap is not None:
            w_freq = w_freq.clamp_max(self.freq_weight_cap)
        if self._collect_frequency_debug:
            zeros_agent = torch.zeros_like(roughness)
            zeros_freq = torch.zeros_like(w_freq)
            zeros_profile = torch.zeros(num_bands, device=z.device, dtype=dtype)
            self._last_frequency_debug = {
                "frequency_credential_mode": self.frequency_credential_mode,
                "w_rough": w_freq.detach(),
                "w_material": w_freq.detach(),
                "mid_profile": zeros_profile.detach(),
                "material_score": zeros_agent.detach(),
                "selective_material": zeros_agent.detach(),
                "material_gate": zeros_agent.detach(),
                "midfreq_shift": zeros_freq.detach(),
                "midfreq_bonus": zeros_freq.detach(),
                "midfreq_alpha": zeros_agent.detach(),
                "midfreq_dominance": zeros_agent.detach(),
                "highfreq_profile": zeros_profile.detach(),
                "highfreq_coherence_shift": zeros_freq.detach(),
            }
        else:
            self._last_frequency_debug = {}
        self._collect_frequency_debug = False
        return w_freq, ell_max, ell_centroids

    def frequency_credentials_all_parameters(
        self,
        z: torch.Tensor,
        num_bands: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_bands != len(self.ell_centroids):
            raise ValueError(
                f"ell_centroids length {len(self.ell_centroids)} does not match "
                f"runtime number of bands {num_bands}"
            )
        if self.roughness_index < 0 or self.roughness_index >= z.shape[-1]:
            raise ValueError(
                f"roughness_index={self.roughness_index} is outside z dimension {z.shape[-1]}"
            )
        if self.use_midfreq_material_boost:
            missing = []
            if self.metallic_index < 0 or self.metallic_index >= z.shape[-1]:
                missing.append(f"metallic_index={self.metallic_index}")
            if self.specular_index < 0 or self.specular_index >= z.shape[-1]:
                missing.append(f"specular_index={self.specular_index}")
            if missing:
                raise ValueError(
                    "mid-frequency material boost requires valid material indices in z; "
                    f"{', '.join(missing)} outside z dimension {z.shape[-1]}"
                )

        roughness_for_ell = z[..., self.roughness_index].to(dtype=dtype).clamp_min(0.0)
        ell_max = (self.freq_c / torch.sqrt(roughness_for_ell + self.eps)).clamp(
            self.ell_min,
            self.ell_cap,
        )
        ell_centroids = torch.tensor(
            self.ell_centroids,
            device=z.device,
            dtype=dtype,
        )
        if self.midfreq_profile is None:
            ell_mid = 10.0
            sigma_mid = 0.45
            mid_profile = torch.exp(
                -0.5
                * (
                    (torch.log(ell_centroids + self.eps) - math.log(ell_mid))
                    / sigma_mid
                ).square()
            )
        else:
            mid_profile = torch.tensor(
                self.midfreq_profile,
                device=z.device,
                dtype=dtype,
            )
        if mid_profile.numel() != num_bands:
            raise ValueError(
                f"midfreq_profile length {mid_profile.numel()} must match "
                f"runtime number of bands {num_bands}"
            )
        if self.midfreq_highfreq_profile is None:
            highfreq_profile = torch.zeros(num_bands, device=z.device, dtype=dtype)
        else:
            highfreq_profile = torch.tensor(
                self.midfreq_highfreq_profile,
                device=z.device,
                dtype=dtype,
            )
        if highfreq_profile.numel() != num_bands:
            raise ValueError(
                f"midfreq_highfreq_profile length {highfreq_profile.numel()} must match "
                f"runtime number of bands {num_bands}"
            )

        w_rough = torch.sigmoid(self.freq_alpha * (ell_max[..., None] - ell_centroids))
        material_gate = torch.zeros_like(roughness_for_ell)
        material_score = torch.zeros_like(roughness_for_ell)
        selective_material = torch.zeros_like(roughness_for_ell)
        midfreq_shift = torch.zeros_like(w_rough)
        midfreq_bonus = torch.zeros_like(w_rough)
        w_material = w_rough
        if self.use_midfreq_material_boost:
            roughness = z[..., self.roughness_index].to(dtype=dtype).clamp(0.0, 1.0)
            metallic = z[..., self.metallic_index].to(dtype=dtype).clamp(0.0, 1.0)
            specular = z[..., self.specular_index].to(dtype=dtype).clamp(0.0, 1.0)
            metallic_gate = torch.pow(metallic, self.midfreq_material_power)
            specular_gate = torch.pow(specular, self.midfreq_material_power)
            material_score = (
                metallic_gate
                + self.midfreq_specular_scale * (1.0 - metallic) * specular_gate
            )
            selective_material = (
                (material_score - self.midfreq_material_threshold)
                / (1.0 - self.midfreq_material_threshold)
            ).clamp(0.0, 1.0)
            material_gate = (
                selective_material
                * torch.pow(1.0 - roughness, self.midfreq_roughness_power)
            )
            midfreq_shift = (
                self.midfreq_boost_weight
                * material_gate[..., None]
                * mid_profile[None, None, :]
            )
            w_material = torch.sigmoid(
                self.freq_alpha
                * (ell_max[..., None] + midfreq_shift - ell_centroids)
            )
            midfreq_bonus = (w_material - w_rough).clamp_min(0.0)

        w_freq = (w_rough + midfreq_bonus).clamp_min(torch.finfo(dtype).tiny)
        midfreq_alpha = torch.zeros_like(roughness_for_ell)
        midfreq_dominance = torch.zeros_like(roughness_for_ell)
        highfreq_coherence_shift = torch.zeros_like(w_freq)
        if self.use_midfreq_highfreq_coherence and num_bands > 2 and z.shape[1] > 1:
            freq_sum = w_freq.sum(dim=1).clamp_min(torch.finfo(dtype).tiny)
            midfreq_alpha = w_freq[:, :, 2] / freq_sum[:, 2, None]
            fair_share = 1.0 / float(z.shape[1])
            midfreq_dominance = (
                (midfreq_alpha - fair_share) / (1.0 - fair_share)
            ).clamp(0.0, 1.0)
            highfreq_coherence_shift = (
                self.midfreq_highfreq_coherence_weight
                * midfreq_dominance[..., None]
                * highfreq_profile[None, None, :]
            )
            w_freq_safe = w_freq.clamp(
                torch.finfo(dtype).tiny,
                1.0 - torch.finfo(dtype).eps,
            )
            w_freq = torch.sigmoid(
                torch.logit(w_freq_safe) + highfreq_coherence_shift
            )
            w_freq = w_freq.clamp_min(torch.finfo(dtype).tiny)
        if self.freq_weight_cap is not None:
            w_freq = w_freq.clamp_max(self.freq_weight_cap)
        if self._collect_frequency_debug:
            self._last_frequency_debug = {
                "frequency_credential_mode": self.frequency_credential_mode,
                "w_rough": w_rough.detach(),
                "w_material": w_material.detach(),
                "mid_profile": mid_profile.detach(),
                "material_score": material_score.detach(),
                "selective_material": selective_material.detach(),
                "material_gate": material_gate.detach(),
                "midfreq_shift": midfreq_shift.detach(),
                "midfreq_bonus": midfreq_bonus.detach(),
                "midfreq_alpha": midfreq_alpha.detach(),
                "midfreq_dominance": midfreq_dominance.detach(),
                "highfreq_profile": highfreq_profile.detach(),
                "highfreq_coherence_shift": highfreq_coherence_shift.detach(),
            }
        else:
            self._last_frequency_debug = {}
        self._collect_frequency_debug = False
        return w_freq, ell_max, ell_centroids

    def spatial_credentials(
        self,
        envmaps: torch.Tensor,
        z: torch.Tensor,
        normal_map: Optional[torch.Tensor],
        normal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, M, _, H, W = envmaps.shape
        sigma_spat_rad = self._sigma_spat_rad(z, envmaps.dtype)
        if normal_map is None:
            self._last_spatial_debug = {
                "sigma_spat_rad": sigma_spat_rad,
                "num_visible_normals": torch.zeros(B, M, device=envmaps.device, dtype=envmaps.dtype),
                "num_used_normals": torch.zeros(B, M, device=envmaps.device, dtype=envmaps.dtype),
                "observed_support": torch.ones(B, M, 1, H, W, device=envmaps.device, dtype=envmaps.dtype),
            }
            return torch.ones(B, M, 1, H, W, device=envmaps.device, dtype=envmaps.dtype)
        # Pipeline convention: an explicitly all-white mask is the sentinel
        # for full visual/angular support. Spatial weighting is neutral then.
        if normal_mask is not None and bool((normal_mask > 0.5).all()):
            num_visible = torch.full(
                (B, M),
                normal_mask.shape[-2] * normal_mask.shape[-1],
                device=envmaps.device,
                dtype=envmaps.dtype,
            )
            self._last_spatial_debug = {
                "sigma_spat_rad": sigma_spat_rad,
                "num_visible_normals": num_visible,
                "num_used_normals": num_visible,
                "full_support_bypass": True,
                "observed_support": torch.ones(B, M, 1, H, W, device=envmaps.device, dtype=envmaps.dtype),
            }
            return torch.ones(B, M, 1, H, W, device=envmaps.device, dtype=envmaps.dtype)
        return self._geodesic_spatial_credentials(envmaps, z, normal_map, normal_mask)

    def _geodesic_spatial_credentials(
        self,
        envmaps: torch.Tensor,
        z: torch.Tensor,
        normal_map: torch.Tensor,
        normal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build a soft union of observed incident-direction supports.

        Every reflected direction seeds a unit-height geodesic Gaussian.  A
        pointwise maximum (rather than a density sum) makes direct observations
        maximally credible, keeps repeated image-space normals from changing the
        result, and lets roughness only dilate credibility into nearby unseen
        directions.
        """
        B, M, _, H, W = envmaps.shape
        device = envmaps.device
        dtype = envmaps.dtype
        normals = self._normal_map_to_bm3hw(normal_map, envmaps)
        if normal_mask is None:
            mask = torch.ones(B, M, 1, normals.shape[-2], normals.shape[-1], device=device, dtype=dtype)
        else:
            mask = normal_mask.to(device=device, dtype=dtype)
            if mask.ndim == 4:
                mask = mask[:, :, None]
            if mask.ndim != 5 or mask.shape[2] != 1:
                raise ValueError(f"normal_mask must be (B,M,H,W) or (B,M,1,H,W), got {tuple(normal_mask.shape)}")

        sigma_spat_rad = self._sigma_spat_rad(z, dtype)
        reflected = self._reflected_directions(normals)
        reflected = reflected.permute(0, 1, 3, 4, 2).reshape(B, M, -1, 3)
        mask_flat = mask.reshape(B, M, -1)

        supports: list[torch.Tensor] = []
        observed_supports: list[torch.Tensor] = []
        num_visible = torch.zeros(B, M, device=device, dtype=dtype)
        num_used = torch.zeros(B, M, device=device, dtype=dtype)
        for b in range(B):
            agent_supports: list[torch.Tensor] = []
            agent_observed_supports: list[torch.Tensor] = []
            for m in range(M):
                visible = mask_flat[b, m] > 0
                refl = reflected[b, m, visible]
                visible_weights = mask_flat[b, m, visible]
                count = refl.shape[0]
                if count == 0:
                    agent_supports.append(torch.zeros(1, 1, 1, H, W, device=device, dtype=dtype))
                    agent_observed_supports.append(torch.zeros(1, 1, 1, H, W, device=device, dtype=dtype))
                    continue

                num_visible[b, m] = count
                num_used[b, m] = count
                sigma = sigma_spat_rad[b, m].clamp_min(self.eps)
                radius = (self.spatial_truncation_sigma * sigma).clamp_max(math.pi)

                # DRMNet equirectangular convention: +Y is the zenith and
                # increasing columns follow alpha=atan2(x, -z) in [0, 2*pi).
                theta0 = torch.acos(refl[:, 1].clamp(-1.0 + self.eps, 1.0 - self.eps))
                alpha0 = torch.remainder(torch.atan2(refl[:, 0], -refl[:, 2]), 2.0 * math.pi)
                iy0 = torch.floor(theta0 * (H / math.pi)).long().clamp(0, H - 1)
                ix0 = torch.floor(alpha0 * (W / (2.0 * math.pi))).long().remainder(W)

                theta_half = max(
                    1,
                    int(math.ceil(float(radius.detach().cpu()) * H / math.pi)) + 1,
                )
                reaches_pole = (theta0 <= radius) | ((math.pi - theta0) <= radius)
                phi_radius = torch.asin(
                    (torch.sin(radius) / torch.sin(theta0).clamp_min(self.eps)).clamp(0.0, 1.0)
                )
                phi_radius = torch.where(reaches_pole, torch.full_like(phi_radius, math.pi), phi_radius)
                phi_half = (
                    torch.ceil(phi_radius * (W / (2.0 * math.pi))).long() + 1
                ).clamp(min=1, max=max(1, W // 2))
                if self.max_phi_window is not None:
                    phi_half = phi_half.clamp(max=min(self.max_phi_window, max(1, W // 2)))

                # Sorting groups similar latitude windows together. This keeps
                # each vectorized chunk close to its true local support size.
                order = torch.argsort(phi_half)
                visible_weights = visible_weights[order]
                theta0 = theta0[order]
                alpha0 = alpha0[order]
                iy0 = iy0[order]
                ix0 = ix0[order]
                phi_half = phi_half[order]
                phi_half_cpu = phi_half.detach().cpu()

                y_offsets = torch.arange(-theta_half, theta_half + 1, device=device)
                kernel_max = torch.zeros(H * W, device=device, dtype=dtype)
                for start in range(0, count, self.normal_chunk_size):
                    end = min(start + self.normal_chunk_size, count)
                    weight_chunk = visible_weights[start:end]
                    theta_chunk = theta0[start:end]
                    alpha_chunk = alpha0[start:end]
                    iy_chunk = iy0[start:end]
                    ix_chunk = ix0[start:end]
                    phi_half_chunk = phi_half[start:end]
                    chunk_size = end - start

                    y_unclamped = iy_chunk[:, None] + y_offsets[None]
                    y_valid = (y_unclamped >= 0) & (y_unclamped < H)
                    y_idx = y_unclamped.clamp(0, H - 1)

                    max_phi_half = int(phi_half_cpu[end - 1].item())
                    if max_phi_half >= W // 2:
                        x_idx = torch.arange(W, device=device)[None].expand(chunk_size, -1)
                        circular_delta = torch.remainder(
                            x_idx - ix_chunk[:, None] + W // 2,
                            W,
                        ) - W // 2
                        x_valid = circular_delta.abs() <= phi_half_chunk[:, None]
                    else:
                        x_offsets = torch.arange(-max_phi_half, max_phi_half + 1, device=device)
                        x_idx = torch.remainder(ix_chunk[:, None] + x_offsets[None], W)
                        x_valid = x_offsets.abs()[None] <= phi_half_chunk[:, None]

                    theta_local = (y_idx.to(dtype) + 0.5) * (math.pi / H)
                    alpha_local = (x_idx.to(dtype) + 0.5) * (2.0 * math.pi / W)
                    dots = (
                        torch.cos(theta_local)[:, :, None] * torch.cos(theta_chunk)[:, None, None]
                        + torch.sin(theta_local)[:, :, None]
                        * torch.sin(theta_chunk)[:, None, None]
                        * torch.cos(alpha_local[:, None, :] - alpha_chunk[:, None, None])
                    )
                    dist = torch.acos(dots.clamp(-1.0 + self.eps, 1.0 - self.eps))
                    local_valid = y_valid[:, :, None] & x_valid[:, None, :] & (dist <= radius)
                    kernel = torch.exp(-0.5 * (dist / sigma).square())
                    source = torch.where(local_valid, kernel, torch.zeros_like(kernel))
                    source = source * weight_chunk[:, None, None]
                    linear_idx = y_idx[:, :, None] * W + x_idx[:, None, :]
                    chunk_max = torch.zeros(H * W, device=device, dtype=dtype).scatter_reduce(
                        0,
                        linear_idx.reshape(-1),
                        source.reshape(-1),
                        reduce="amax",
                        include_self=True,
                    )
                    kernel_max = torch.maximum(kernel_max, chunk_max)
                # The continuous reflected direction generally falls between
                # equirectangular pixel centers, so the sampled Gaussian peak
                # would otherwise be slightly below one. Mark its nearest
                # discrete cell explicitly: directly observed support is
                # exactly 1, while only unseen neighbours are roughness-dilated.
                seed_idx = iy0 * W + ix0
                seed_values = visible_weights.clamp(0.0, 1.0)
                seed_max = torch.zeros(H * W, device=device, dtype=dtype).scatter_reduce(
                    0,
                    seed_idx,
                    seed_values,
                    reduce="amax",
                    include_self=True,
                )
                kernel_max = torch.maximum(kernel_max, seed_max)
                support = kernel_max.reshape(1, 1, 1, H, W).clamp(0.0, 1.0)
                agent_supports.append(support)
                agent_observed_supports.append(seed_max.reshape(1, 1, 1, H, W))
            supports.append(torch.cat(agent_supports, dim=1))
            observed_supports.append(torch.cat(agent_observed_supports, dim=1))

        self._last_spatial_debug = {
            "sigma_spat_rad": sigma_spat_rad,
            "num_visible_normals": num_visible,
            "num_used_normals": num_used,
            "observed_support": torch.cat(observed_supports, dim=0),
        }
        return torch.cat(supports, dim=0)

    def spherical_gaussian_filter(self, values: torch.Tensor, sigma: float) -> torch.Tensor:
        """Area-weighted Gaussian filter for equirectangular sphere maps."""
        if values.ndim < 4:
            raise ValueError(f"expected tensor with spatial H,W dims, got {tuple(values.shape)}")

        original_shape = values.shape
        H, W = original_shape[-2:]
        x = values.reshape(-1, 1, H, W)
        theta = (torch.arange(H, device=values.device, dtype=values.dtype) + 0.5) * (math.pi / H)
        sin_theta = torch.sin(theta).clamp_min(self.eps).view(1, 1, H, 1)
        weighted = x * sin_theta

        num = self._separable_gaussian_2d(weighted, sigma)
        den = self._separable_gaussian_2d(sin_theta.expand_as(x[:, :1]), sigma)
        filtered = num / den.clamp_min(self.eps)
        return filtered.reshape(original_shape)

    def _separable_gaussian_2d(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel = torch.exp(-0.5 * (coords / float(sigma)) ** 2)
        kernel = kernel / kernel.sum().clamp_min(self.eps)

        x = F.pad(x, (radius, radius, 0, 0), mode="circular")
        x = F.conv2d(x, kernel.view(1, 1, 1, -1))
        x = F.pad(x, (0, 0, radius, radius), mode="replicate")
        x = F.conv2d(x, kernel.view(1, 1, -1, 1))
        return x

    def _sigma_spat_rad(self, z: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        roughness = z[..., self.roughness_index].to(dtype=dtype).clamp(0.0, 1.0)
        sigma_min = torch.as_tensor(
            self.sigma_min_deg * math.pi / 180.0,
            device=z.device,
            dtype=dtype,
        )
        sigma_max = torch.as_tensor(
            self.sigma_max_deg * math.pi / 180.0,
            device=z.device,
            dtype=dtype,
        )
        return sigma_min + (sigma_max - sigma_min) * roughness

    def _reflected_directions(self, normals: torch.Tensor) -> torch.Tensor:
        wo = torch.tensor([0.0, 0.0, 1.0], device=normals.device, dtype=normals.dtype).view(1, 1, 3, 1, 1)
        dots = (wo * normals).sum(dim=2, keepdim=True)
        return F.normalize(2.0 * dots * normals - wo, dim=2, eps=self.eps)

    def _normal_map_to_bm3hw(self, normal_map: torch.Tensor, envmaps: torch.Tensor) -> torch.Tensor:
        normals = normal_map.to(device=envmaps.device, dtype=envmaps.dtype)
        if normals.ndim != 5:
            raise ValueError(f"normal_map must be 5-D, got {tuple(normal_map.shape)}")
        if normals.shape[2] == 3:
            return F.normalize(normals, dim=2, eps=self.eps)
        if normals.shape[-1] == 3:
            normals = normals.permute(0, 1, 4, 2, 3).contiguous()
            return F.normalize(normals, dim=2, eps=self.eps)
        raise ValueError(f"normal_map must be (B,M,3,H,W) or (B,M,H,W,3), got {tuple(normal_map.shape)}")
