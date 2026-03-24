"""
FakeScoreEvaluator
==================
Wraps a *copy* of the Wan2.2 DiT (WanModel) as the "fake score" network used
in DMD-style (Distribution Matching Distillation) training.

Design constraints
------------------
* WanModel is **never modified** – we only hold a reference and call it.
* This module owns its own AdamW optimizer so the caller can run two-timescale
  updates:  fake-score lr >> student lr.
* The fake score is trained to track the *student* distribution, so its
  optimizer is updated on student-generated samples exclusively.
* The real score (frozen teacher) stays outside this module – the caller just
  passes in the teacher's velocity prediction directly.

DMD gradient signal
-------------------
The per-sample distribution-matching loss on the student output  x_gen  is:

    L_dm  =  stopgrad( v_real(x_noisy, σ) − v_fake(x_noisy, σ) ) · x_gen

where  x_noisy = x_gen + σ · ε,  and the gradient flows only through  x_gen.
This is computed in train_distill.py; this module only handles forward passes
and fake-score weight updates.
"""

import copy
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from diffsynth.models.wan_video_dit import WanModel

logger = logging.getLogger(__name__)


class FakeScoreEvaluator(nn.Module):
    """
    Fake score network for DMD-style distillation.

    Parameters
    ----------
    dit_state_dict : dict
        State dict of the *teacher* WanModel to initialise weights from.
        Typically  pipe.dit.state_dict().
    dit_config : dict
        Keyword-args used to construct a new WanModel instance.
        Must include all required constructor parameters
        (in_dim, dim, ffn_dim, freq_dim, text_dim, out_dim,
         num_heads, num_layers, patch_size, …).
    lr : float
        Learning rate for the fake-score AdamW optimizer.
        Should be *higher* than the student lr (e.g. 1e-4 vs 1e-5).
    weight_decay : float
        AdamW weight decay.
    device : torch.device | str
    torch_dtype : torch.dtype
    lora_rank : int or None
        If not None, only LoRA adapters are created and trained in the fake
        score (saves memory).  0 / None → full parameter fine-tuning.
    """

    def __init__(
        self,
        dit_state_dict: Dict[str, torch.Tensor],
        dit_config: Dict[str, Any],
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        device: torch.device = torch.device("cuda"),
        torch_dtype: torch.dtype = torch.bfloat16,
        lora_rank: Optional[int] = None,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

        # Build a fresh WanModel instance with the same architecture
        logger.info("[FakeScoreEvaluator] Building fake-score DiT …")
        self.dit = WanModel(**dit_config)
        self.dit.load_state_dict(dit_state_dict, strict=True)
        self.dit = self.dit.to(device=self.device, dtype=self.torch_dtype)

        if lora_rank and lora_rank > 0:
            self._inject_lora(lora_rank)
            trainable_params = [p for p in self.dit.parameters() if p.requires_grad]
            logger.info(f"[FakeScoreEvaluator] LoRA rank={lora_rank}: "
                        f"{sum(p.numel() for p in trainable_params):,} trainable params")
        else:
            # Full fine-tuning: all parameters are trainable
            for p in self.dit.parameters():
                p.requires_grad_(True)
            trainable_params = list(self.dit.parameters())
            logger.info(f"[FakeScoreEvaluator] Full fine-tune: "
                        f"{sum(p.numel() for p in trainable_params):,} params")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.lr = lr

    # ------------------------------------------------------------------
    # Forward  (velocity prediction under fake-score distribution)
    # ------------------------------------------------------------------

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict velocity v_fake(x_noisy, t) for a batch of noisy latents.

        The forward pass is done **with gradients** so that the fake-score
        optimizer can back-prop through it when updating fake-score weights.
        To compute the stop-gradient term in the student loss, call this inside
        a  torch.no_grad()  context or use  .detach()  on the result.

        Args:
            latents:      [B, C, T, H, W]  noisy latents
            timestep:     scalar or [B]  tensor (in scheduler units)
            context:      [B, L, D]  text embeddings
            clip_feature: optional image feature for I2V
            y:            optional reference frames
            **kwargs:     passed through to WanModel.forward

        Returns:
            v_fake: [B, C, T, H, W]  velocity prediction
        """
        return self.dit(
            x=latents,
            t=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Fake-score weight update  (two-timescale step)
    # ------------------------------------------------------------------

    def update_step(
        self,
        loss_fake_score: torch.Tensor,
    ) -> float:
        """
        Perform one gradient-descent step on the fake-score network.

        Call this *after* computing the fake-score training loss:
            L_fs = || v_fake(x_gen_noisy, σ) − stopgrad(v_fake_target) ||²
        where the target is derived from the student-generated samples.

        Args:
            loss_fake_score: scalar loss tensor (with grad_fn).

        Returns:
            loss value as a Python float.
        """
        self.optimizer.zero_grad()
        loss_fake_score.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.dit.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        return loss_fake_score.item()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def get_velocity(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        no_grad: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Helper that optionally wraps forward() in torch.no_grad().

        Use no_grad=True when computing the stop-gradient term in the
        student DMD loss (avoids storing the fake-score computation graph).
        Use no_grad=False when updating the fake-score weights.
        """
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            return self.forward(latents, timestep, context,
                                clip_feature=clip_feature, y=y, **kwargs)

    @classmethod
    def from_pipeline(
        cls,
        pipe,  # PanoramaWanPipeline
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        lora_rank: Optional[int] = None,
    ) -> "FakeScoreEvaluator":
        """
        Convenience constructor: initialise the fake score from an existing
        pipeline's DiT weights and config.

        Args:
            pipe:  A PanoramaWanPipeline (or WanVideoPipeline) instance whose
                   .dit attribute is the teacher WanModel.
        """
        dit = pipe.dit
        # Gather constructor config from the model's stored attributes
        dit_config = dict(
            in_dim=dit.in_dim,
            dim=dit.dim,
            ffn_dim=dit.ffn_dim,
            freq_dim=dit.freq_dim,
            text_dim=dit.text_dim,
            out_dim=dit.out_dim,
            num_heads=dit.num_heads,
            num_layers=dit.num_layers,
            patch_size=dit.patch_size,
            window_size=getattr(dit, "window_size", (-1, -1)),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        )
        return cls(
            dit_state_dict=dit.state_dict(),
            dit_config=dit_config,
            lr=lr,
            weight_decay=weight_decay,
            device=pipe.device,
            torch_dtype=pipe.torch_dtype,
            lora_rank=lora_rank,
        )

    # ------------------------------------------------------------------
    # LoRA injection (optional, memory-efficient fake-score training)
    # ------------------------------------------------------------------

    def _inject_lora(self, rank: int):
        """
        Replace q/k/v/o projections in all DiT attention blocks with LoRA
        versions.  Only LoRA parameters are set requires_grad=True.
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            target_modules = ["q", "k", "v", "o"]
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=target_modules,
                lora_dropout=0.0,
                bias="none",
            )
            self.dit = get_peft_model(self.dit, lora_config)
            logger.info(f"[FakeScoreEvaluator] Injected LoRA via peft (rank={rank})")
        except ImportError:
            logger.warning(
                "[FakeScoreEvaluator] peft not installed – falling back to "
                "manual LoRA injection."
            )
            self._inject_lora_manual(rank)

    def _inject_lora_manual(self, rank: int):
        """
        Minimal manual LoRA without peft dependency.
        Wraps Linear layers named 'q', 'k', 'v', 'o' inside each DiT block's
        self_attn with a LoRA adapter.
        """
        for name, module in self.dit.named_modules():
            if not name.endswith((".q", ".k", ".v", ".o")):
                continue
            if not isinstance(module, nn.Linear):
                continue
            in_f, out_f = module.in_features, module.out_features
            # Freeze original weights
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)
            # Attach LoRA A/B as extra parameters
            lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
            lora_B = nn.Parameter(torch.zeros(out_f, rank))
            module.register_parameter("lora_A", lora_A)
            module.register_parameter("lora_B", lora_B)
            # Patch forward
            original_forward = module.forward

            def _lora_forward(x, _A=lora_A, _B=lora_B, _orig=original_forward):
                return _orig(x) + (x @ _A.T) @ _B.T

            module.forward = _lora_forward
