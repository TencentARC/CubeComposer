"""
DistillFlowMatchScheduler
=========================
Extends diffsynth's FlowMatchScheduler with helpers needed for distillation:

  * get_ode_pair(t1_id, t2_id)
      Returns the pair of sigma values at two timestep indices, used to define
      consistency targets on the ODE trajectory.

  * velocity_to_score(v_pred, sigma)
      Converts a velocity-prediction output  v = (ε - x0)  to a score-function
      estimate  ∇logp ≈ -v/σ ,  needed to compute the DMD distribution-matching
      gradient.

  * sample_distill_timestep(mode, device)
      Convenience sampler that picks a single timestep id according to the
      chosen distillation strategy ('uniform', 'logit_normal', 'boundary').

All other scheduler behaviour (add_noise, step, training_weight, …) is
inherited unchanged from FlowMatchScheduler.  The diffsynth source is never
modified.
"""

import torch
import math
from typing import Literal, Tuple

# Import from the *local* diffsynth copy (project root takes priority in sys.path)
from diffsynth.schedulers.flow_match import FlowMatchScheduler


class DistillFlowMatchScheduler(FlowMatchScheduler):
    """
    Drop-in replacement for FlowMatchScheduler that adds distillation helpers.

    Usage
    -----
    # Construct exactly like the base class:
    scheduler = DistillFlowMatchScheduler(
        num_inference_steps=50,
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
    )
    # Must call set_timesteps (with training=True) before using training helpers:
    scheduler.set_timesteps(num_inference_steps, training=True)
    """

    # ------------------------------------------------------------------
    # ODE-pair sampling for consistency distillation
    # ------------------------------------------------------------------

    def get_ode_pair(
        self,
        t1_id: int,
        t2_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the sigma values at two timestep indices.

        In the consistency distillation loss we require the student to map
        x_{σ1}  directly to the clean prediction that a full multi-step ODE
        would give starting from  x_{σ2} (where σ1 > σ2, i.e. t1_id < t2_id
        in the decreasing-sigma schedule).

        Args:
            t1_id: Index of the *noisier* timestep  (smaller index = larger σ).
            t2_id: Index of the *cleaner* timestep  (larger  index = smaller σ).

        Returns:
            (sigma1, sigma2)  – scalar tensors on CPU.
        """
        assert 0 <= t1_id < len(self.sigmas), f"t1_id {t1_id} out of range"
        assert 0 <= t2_id < len(self.sigmas), f"t2_id {t2_id} out of range"
        return self.sigmas[t1_id], self.sigmas[t2_id]

    def sample_ode_pair(
        self,
        min_gap: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a (t1_id, t2_id) pair with t1_id < t2_id (noisier first)
        and a gap of at least `min_gap` steps.

        Returns:
            t1_id, t2_id, sigma1, sigma2
        """
        n = len(self.sigmas)
        t1_id = torch.randint(0, n - min_gap, (1,)).item()
        t2_id = torch.randint(t1_id + min_gap, n, (1,)).item()
        sigma1, sigma2 = self.get_ode_pair(t1_id, t2_id)
        return t1_id, t2_id, sigma1.to(device), sigma2.to(device)

    # ------------------------------------------------------------------
    # Score-function conversion  (velocity → score)
    # ------------------------------------------------------------------

    def velocity_to_score(
        self,
        v_pred: torch.Tensor,
        sigma: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Convert a velocity-prediction  v = ε − x0  to a score estimate.

        Under flow matching the forward process is
            x_t = (1 − σ) x0 + σ ε
        and the velocity field satisfies  v_θ(x_t, t) = ε − x0.
        The score function is  ∇_{x_t} log p_t(x_t) = −(x_t − (1−σ)x̂0) / σ²
        which, after substituting  x̂0 = x_t − σ·v_θ,  simplifies to

            ∇ log p ≈ −v_θ / σ

        Args:
            v_pred: velocity prediction from the DiT, shape [B, C, T, H, W] or any.
            sigma:  scalar or broadcastable tensor, the noise level.

        Returns:
            score estimate, same shape as v_pred.
        """
        return -v_pred / (sigma + eps)

    # ------------------------------------------------------------------
    # Convenience: single-timestep sampler for distillation
    # ------------------------------------------------------------------

    def sample_distill_timestep(
        self,
        mode: Literal["uniform", "logit_normal", "boundary"] = "logit_normal",
        device: torch.device = torch.device("cpu"),
        min_id: int = 0,
        max_id: int = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample a single timestep id and return (timestep_id, timestep, sigma).

        Modes
        -----
        uniform       – uniform over [min_id, max_id)
        logit_normal  – logit-normal, concentrates samples away from extremes
                        (good for DMD where mid-noise is most informative)
        boundary      – always returns timestep 0 (the single denoising step
                        used in one-step generation)

        Returns:
            (timestep_id: int,  timestep: scalar tensor,  sigma: scalar tensor)
        """
        n = len(self.sigmas) if max_id is None else max_id

        if mode == "boundary":
            tid = 0
        elif mode == "uniform":
            tid = torch.randint(min_id, n, (1,)).item()
        elif mode == "logit_normal":
            # Sample u ~ logit-Normal(0, 1), map to [min_id, n)
            u = torch.randn(1).item()         # standard normal
            u = 1.0 / (1.0 + math.exp(-u))   # sigmoid → (0,1)
            tid = min_id + int(u * (n - min_id - 1))
            tid = max(min_id, min(n - 1, tid))
        else:
            raise ValueError(f"Unknown distill timestep mode: {mode}")

        timestep = self.timesteps[tid].to(device)
        sigma = self.sigmas[tid].to(device)
        return tid, timestep, sigma

    # ------------------------------------------------------------------
    # One-step denoising helper  (used in training_rollout)
    # ------------------------------------------------------------------

    def one_step_denoise(
        self,
        x_noisy: torch.Tensor,
        v_pred: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a single ODE step from sigma → 0.

        x0_pred = x_noisy − σ · v_pred
            (derived from  x_t = (1−σ)x0 + σ·ε  and  v = ε − x0)

        Args:
            x_noisy: noisy latent [B, C, T, H, W]
            v_pred:  velocity prediction from the model, same shape
            sigma:   scalar noise level

        Returns:
            x0_pred: predicted clean latent, same shape as x_noisy
        """
        return x_noisy - sigma * v_pred
