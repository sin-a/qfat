import itertools
import logging
import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Optional, Tuple, Union

import hydra
import numpy as np
import torch
import torch.nn as nn
from line_profiler import profile
from numpy.typing import NDArray
from torch.distributions import MixtureSameFamily, MultivariateNormal

from qfat.conf.configs import DecoderBlockCfg, EncoderCfg, OptimizerCfg
from qfat.datasets.dataset import Batch
from qfat.models.decoder_block import DecoderBlock
from qfat.models.generative_model import (
    GenerativeModel,
    ModelOutput,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchSequenceGMMParams:
    """Defines the return type of the givt forward."""

    variances: (
        torch.Tensor
    )  # variance diagonals (batch_size, sequence_length, kmixtures, out_dim)
    locs: torch.Tensor  # means (batch_size, sequence_length, kmixtures, out_dim)
    mixture_probs: torch.Tensor  # (batch_size, sequence_length, kmixtures)

    def __getitem__(self, idx) -> "BatchSequenceGMMParams":
        """Allows indexing across all tensor attributes simultaneously."""
        field_values = {f.name: getattr(self, f.name)[idx] for f in fields(self)}
        return BatchSequenceGMMParams(**field_values)

    def with_grad(self) -> "BatchSequenceGMMParams":
        """
        Returns a new BatchSequenceGMMParams with .requires_grad_(True)
        set on each field. Leaves the current instance immutable.
        """
        new_variances = self.variances.clone().requires_grad_()
        new_locs = self.locs.clone().requires_grad_()
        new_mixture_probs = self.mixture_probs.clone().requires_grad_()
        return BatchSequenceGMMParams(
            variances=new_variances, locs=new_locs, mixture_probs=new_mixture_probs
        )


class MultivariateNormalDiag(MultivariateNormal):
    """A faster implementation of log prob of multivariate normal with diagonal covariances"""

    def __init__(self, loc, variances, validate_args=None):
        self.variances = variances
        super(MultivariateNormalDiag, self).__init__(
            loc,
            scale_tril=torch.diag_embed(variances**0.5),
            validate_args=validate_args,
        )

    def log_prob(self, value):
        diff = value - self.loc
        M = torch.sum(diff**2 / self.variances, dim=-1)
        log_det = self.variances.log().sum(-1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M + log_det)


class QFAT(GenerativeModel):
    """Implements a simplified infinite vocabulary transformer, without a VAE.
    Reference: https://arxiv.org/pdf/2312.02116
    """

    def __init__(
        self,
        context_len: int,
        input_dim: int,
        n_layer: int,
        out_dim: int,
        mixture_size: int,
        embd_dropout: float,
        encoder_cfg: EncoderCfg,
        decoder_block_cfg: DecoderBlockCfg,
        optimizer_cfg: OptimizerCfg,
        variance_tol: float = 1e-6,
        masking_strategy: Literal["causal"] = "causal",
        full_covariance: bool = False,
        lambda_mixtures: float = 0,
        pad_sampling: bool = False,
        conditional_seq_dim: Optional[int] = None,
        share_projection_layer: bool = True,
        share_state_encoder: bool = True,
        sample_fn: Literal["gmm", "modes"] = "modes",
        goal_pos_emb: bool = False,
        history_mask_prob: float = 0,
        **kwargs,
    ):
        """Instantiates a QFAT model, based on the minGPT backbone.

        Args:
            context_len (int): Maximum sequence length to have at the input.
            input_dim (int): Dimensionality of the input.
            n_layer (int): Number of decoder blocks to apply.
            out_dim (int): Output space dimension.
            mixture_size (int): Number of Gaussian Mixtures per token.
            embd_dropout (float): Probability of dropout after initial projection of the input.
            decoder_block_cfg (DecoderBlockCfg): Cfguration of the decoder.
            variance_tol (float, optional): Lower bound on the variances of the mixtures. Defaults to 1e-8.
            masking_strategy (Literal['causal']): Masking strategy to
                apply in the decoder blocks. Defaults to "causal".
            full_covariance (bool): Whether to use a full covariance matrix or not.
                If false, a diagonla covariance will be used.
            lambda_mixtures (float): The entropy regularizaion weight on the mixture probabilities.
                Defaults to 0.
            pad_sampling (bool, optional): Whether to pre-pad the context with zeros when samplig from the model.
                Defaults to False.
            conditional_seq_dim (Optional[int]): The dimensionality of the sequence used for conditioning.
            share_projection_layer (bool, optional): Wether to share the linear projection layer of the states with
                the conditional sequence. Defaults to True.
            share_state_encoder (bool, optional): Wether to share the state encoder with the conditional sequence.
                Defaults to True.
            sample_fn: (Literal["gmm", "modes"]): Whether to sample from the gmm or only sample the GMM modes.
            goal_pos_emb (bool, optional): Whether or not to add a positional embedding to the conditional sequence.
                Defaults to False.

        Raises:
            ValueError: If the state encoder is not shared and the model uses a conditional sequence.
            ValueError: If the projection layer should be shared between the state and the conditional sequence but
                their dimensionality does not match.
        """
        super().__init__()
        self.context_len = context_len
        self.input_dim = input_dim
        self.n_layer = n_layer
        self.out_dim = out_dim
        self.mixture_size = mixture_size
        self.embd_dropout = embd_dropout
        self.decoder_block_cfg = decoder_block_cfg
        self.optimizer_cfg = optimizer_cfg
        self.variance_tol = variance_tol
        self.masking_strategy = masking_strategy
        self.full_covariance = full_covariance
        self.device = "cpu"  # will get updated if 'to' is called
        self.lambda_mixtures = lambda_mixtures
        self.pad_sampling = pad_sampling
        self.conditional_seq_dim = conditional_seq_dim
        self.share_state_encoder = share_state_encoder
        self.share_projection_layer = share_projection_layer
        mha_cfg = decoder_block_cfg.mha_cfg
        self.goal_pos_emb = goal_pos_emb

        self.act_dropout = nn.Dropout(0.5)
        self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.transformer = nn.ModuleDict(
            dict(
                fc_in=nn.Linear(self.input_dim, mha_cfg.embed_dim),
                pos_embed=nn.Embedding(self.context_len, mha_cfg.embed_dim),
                dropout=nn.Dropout(self.embd_dropout),
                dec_blocks=nn.ModuleList(
                    [DecoderBlock(decoder_block_cfg) for _ in range(self.n_layer)]
                ),
                ln_f=nn.LayerNorm(mha_cfg.embed_dim),
            )
        )
        self.conditional_seq_layer = (
            None  # used to embed e.g goal sequences to the same context as the states.
        )
        if self.conditional_seq_dim is not None and not self.share_projection_layer:
            self.conditional_seq_layer = nn.Linear(
                self.conditional_seq_dim, mha_cfg.embed_dim
            )
        elif self.conditional_seq_dim is not None and self.share_projection_layer:
            if self.conditional_seq_dim != self.input_dim:
                raise ValueError(
                    "The conditional sequence and the input have different dimensions. Can't share projection layer."
                )
            self.conditional_seq_layer = self.transformer.fc_in
        else:
            logger.info(
                "Conditonal layer not initialized. Please specify the conditional sequence dimension if you wish to instantiate it."
            )
        if self.goal_pos_emb:
            self.transformer.goal_pos_embed = nn.Embedding(
                self.context_len, mha_cfg.embed_dim
            )
            self.transformer.goal_pos_embed.apply(self._init_weights)
        else:
            self.transformer.goal_pos_embed = None

        if not self.share_state_encoder:
            raise ValueError(
                "It is only supported to share the state encoder with the conditional sequence."
            )
        self.mha_kwargs = self.register_mask(self.masking_strategy)
        self.fc_out = nn.Linear(
            decoder_block_cfg.mha_cfg.embed_dim, self._get_gmm_param_size()
        )
        self.scale_normalization = torch.nn.Softplus()
        self.mixtures_normalization = torch.nn.Softmax(dim=-1)

        self.transformer.apply(self._init_weights)
        self._init_gmm_weights()  # fc_out.apply(self._init_weights)
        if self.conditional_seq_layer is not None:
            self.conditional_seq_layer.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.transformer.parameters())
        logger.info("number of parameters: %.2fM" % (n_params / 1e6,))
        self.history_mask_prob = history_mask_prob
        self.sample_fn = sample_fn

    def _configure_optimizer(self) -> None:
        """Configures the optimizer, excluding encoder parameters from decay/no_decay sets."""
        optim_cfg: OptimizerCfg = self.optimizer_cfg
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (
            torch.nn.Linear,
            torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
        )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        named_modules = itertools.chain(
            self.transformer.named_modules(),
            self.fc_out.named_modules(),
            self.conditional_seq_layer.named_modules()
            if self.conditional_seq_layer is not None
            else [],
        )  # TODO: Add all trainable modules here! Otherwise you will forget something. Exclusion rather than inclusion

        for mn, m in named_modules:
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and (
                    isinstance(m, whitelist_weight_modules)
                    or (
                        pn.endswith("in_proj_weight")
                        and isinstance(m, torch.nn.MultiheadAttention)
                    )
                ):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {
            pn: p
            for pn, p in itertools.chain(
                self.transformer.named_parameters(),
                self.fc_out.named_parameters(),
            )
        }

        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, (
            f"Parameters {inter_params} made it into both decay and no_decay sets!"
        )
        assert len(param_dict.keys() - union_params) == 0, (
            f"Parameters {param_dict.keys() - union_params} were not separated into either decay or no_decay set!"
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": optim_cfg.weight_decay,
                "lr": optim_cfg.learning_rate * 0.1,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
                "lr": optim_cfg.learning_rate,
            },
        ]
        optim_groups.append(
            {
                "params": [
                    p for _, p in self.encoder.named_parameters() if p.requires_grad
                ],
                "lr": optim_cfg.learning_rate * optim_cfg.encoder_lr_fraction,
                "weight_decay": optim_cfg.weight_decay,
            }
        )

        optimizer = torch.optim.AdamW(optim_groups, betas=optim_cfg.betas)
        return optimizer

    @property
    def is_conditional(self) -> bool:
        return self._is_conditional

    @is_conditional.setter
    def is_conditional(self, value):
        raise ValueError()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = next(
            self.parameters()
        ).device  # assumes model params are on the same device, which is fine for small models like mine :(

    def _get_gmm_param_size(self) -> int:
        """Returns the shape of the gmm parametrization.

        If full covariance, the lower traingular matrix is parametrized.
        Otherwise, only the diagonal variances are used.
        """
        means_param_size = self.mixture_size * self.out_dim
        mixtures_param_size = self.mixture_size
        variances_param_size = self.mixture_size * self.out_dim
        if self.full_covariance:
            variances_param_size *= (self.out_dim + 1) / 2
        return int(means_param_size + variances_param_size + mixtures_param_size)

    def _init_gmm_weights(self):
        """
        Initializes the GMM parameters (the final fc_out bias) so that the means are
        spread out using a kmeans++ style algorithm over the space [-1,1]^out_dim, the
        mixture weights are uniform, and the variances are set so that after Softplus they equal 1.
        """
        with torch.no_grad():
            # Zero out the weights so that the output depends solely on the bias.
            self.fc_out.weight.zero_()
            bias = self.fc_out.bias

            # Calculate sizes for the three parameter groups.
            n_means = self.mixture_size * self.out_dim
            n_mixtures = self.mixture_size
            # ------------------------------
            # KMEANS++ initialization for means.
            # ------------------------------
            # We assume the space is normalized in [-1,1]^D.
            N_candidates = 1000  # number of candidate points; adjust as needed.
            candidates = torch.empty(
                (N_candidates, self.out_dim), device=bias.device
            ).uniform_(-1, 1)

            # Randomly pick the first center from the candidate pool.
            first_idx = torch.randint(0, N_candidates, (1,)).item()
            centers = candidates[first_idx].unsqueeze(0)  # shape: (1, out_dim)

            # Select the remaining centers.
            for _ in range(1, self.mixture_size):
                # Compute the squared Euclidean distance of each candidate to the closest center.
                # Using torch.cdist for pairwise distances:
                dists = torch.cdist(candidates, centers, p=2).pow(
                    2
                )  # shape: (N_candidates, current_number_of_centers)
                min_dists, _ = torch.min(dists, dim=1)  # shape: (N_candidates,)

                # Choose the next center with probability proportional to its squared distance.
                probs = min_dists / min_dists.sum()
                next_idx = torch.multinomial(probs, 1).item()
                next_center = candidates[next_idx].unsqueeze(0)  # shape: (1, out_dim)
                centers = torch.cat([centers, next_center], dim=0)

            # Now 'centers' is a tensor of shape (mixture_size, out_dim) containing the chosen means.
            means = centers  # each row is one mean
            # Set the means in the bias (the first block of parameters).
            bias[:n_means].copy_(means.view(-1))

            # ------------------------------
            # Initialize mixture logits.
            # ------------------------------
            # Set all logits to 0 so that after softmax the mixture weights are uniform.
            bias[n_means : n_means + n_mixtures].fill_(0.0)

            # ------------------------------
            # Initialize variances.
            # ------------------------------
            # Choose v such that Softplus(v) = 1. Here, v = ln(e^1 - 1) ≈ 0.5413.
            variance_init = math.log(math.exp(1) - 1)  # ~0.5413
            bias[n_means + n_mixtures :].fill_(variance_init)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes the trainable weights.

        Copied from https://github.com/karpathy/minGPT.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer)
                )

    def register_mask(
        self, masking_strategy: Optional[Literal["causal", "masked_git", "diag"]] = None
    ) -> Dict[str, Any]:
        """Registers a 'mask' attribute to the module as a buffer (i.e not a model parameter).

        The mask is utilized by the decoder blocks in the multiheaded self attention modules.

        Args:
            masking_strategy (Literal['causal'; 'diag']): The masking strategy. Defaults to None.

        Raises:
            NotImplementedError: If the masking strategy is not yet implemented.

        Returns:
            Dict[str, Any]: Extra parameters to the multiheaded attention module forward pass.
            For all options, please check here:
            https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
        """
        if masking_strategy == "causal":
            self.register_buffer(
                "mask",
                torch.tril(
                    torch.ones(self.context_len, self.context_len, dtype=torch.bool)
                ),
            )
            return {"is_causal": True}
        if masking_strategy == "diag":
            self.register_buffer(
                "mask",
                torch.eye(self.context_len, self.context_len, dtype=torch.bool),
            )
            return {"is_causal": True}
        else:
            raise NotImplementedError("Only causal masking is currently supported.")

    def get_gmm_params(self, x: torch.Tensor) -> BatchSequenceGMMParams:
        """Parses the GMM parameters from the input tensor.

        Args:
            x (torch.Tensor): The output of the transformer decoder blocks,
                after layer normalization has been applied to it.

        Returns:
            BatchSequenceGMMParams: A dataclass that wraps the parsed GMM parameters.
        """
        gmm_params = self.fc_out(x)

        locs, mixtures, variances = (
            gmm_params[:, :, : self.mixture_size * self.out_dim],
            gmm_params[
                :,
                :,
                self.mixture_size * self.out_dim : self.mixture_size * self.out_dim
                + self.mixture_size,
            ],
            gmm_params[:, :, self.mixture_size * self.out_dim + self.mixture_size :],
        )
        variances = torch.max(
            self.scale_normalization(variances),
            torch.tensor(self.variance_tol).expand_as(variances).to(x.device),
        )
        mixtures = self.mixtures_normalization(
            mixtures
        )  # (batch_size, sequence_length, mixture_size)

        variances_last_dim = int(
            self.out_dim * (self.out_dim + 1) / 2
            if self.full_covariance
            else self.out_dim
        )
        gmm_params = BatchSequenceGMMParams(
            variances.view(
                *variances.shape[:-1], self.mixture_size, variances_last_dim
            ),
            locs=locs.view(*locs.shape[:-1], self.mixture_size, self.out_dim),
            mixture_probs=mixtures,
        )
        return gmm_params

    def get_distribution(
        self, gmm_params: BatchSequenceGMMParams, last_only: bool = False
    ) -> torch.distributions.Distribution:
        """Returns a torch distribution given the passed GMM parameters.

        Args:
            gmm_params (BatchSequenceGMMParams):  The parameters of the Gaussian Mixture models
              for all batches and all tokens.
            last_only (bool, optional): Whether to only return the distribution of the last element
                of in the sequence. Set to True when auto-regressive sampling.

        Returns:
            torch.distributions.Distribution: A batched, per token GMM distribution.

        """
        if last_only:
            gmm_params = gmm_params[:, -1, ...]

        categorical_dist = torch.distributions.Categorical(
            probs=gmm_params.mixture_probs
        )
        if self.full_covariance:
            variances = torch.zeros(
                *gmm_params.variances.shape[:-1], self.out_dim, self.out_dim
            ).to(gmm_params.variances.device)
            tril_indices = torch.tril_indices(row=self.out_dim, col=self.out_dim)
            variances[..., tril_indices[0], tril_indices[1]] = gmm_params.variances
            gaussian_dist = torch.distributions.MultivariateNormal(
                loc=gmm_params.locs, scale_tril=variances**0.5
            )
        else:
            variances = torch.diag_embed(gmm_params.variances)
            gaussian_dist = MultivariateNormalDiag(
                loc=gmm_params.locs, variances=gmm_params.variances
            )

        dist = MixtureSameFamily(
            mixture_distribution=categorical_dist,
            component_distribution=gaussian_dist,
        )
        return dist

    @staticmethod
    def mask_history_states(
        validity_mask: torch.Tensor, mask_prob: float
    ) -> torch.Tensor:
        """
        Masks historical states. This is done to counter act the effect of causal confusion in imitation learning.
        https://arxiv.org/pdf/1905.11979
        """
        B, T = validity_mask.shape
        if T <= 1:
            return validity_mask

        # Generate a mask for the history tokens: for the first T-1 tokens, sample a Bernoulli variable
        # that is 1 with probability (1 - mask_prob) (i.e., keep the state), and 0 otherwise.
        # The most recent token is always kept.
        history_mask = torch.bernoulli(
            torch.full((B, T - 1), 1 - mask_prob, device=validity_mask.device)
        )
        last_mask = torch.ones(B, 1, device=validity_mask.device)
        mask = torch.cat([history_mask, last_mask], dim=1)  # shape: (B, T)
        return (validity_mask * mask).to(torch.bool)

    @profile
    def forward(self, batch: Batch) -> ModelOutput:
        """Computes the GIVT output and optionally the loss function if a target y is passed.

        Args:
            batch (Batch): Dataclass containing the batched inputs x, outputs y,
                and a validiting mask for each input token.

        Returns:
            ModelOutput: Dataclass containing the predicted sequence of GMM params,
                and an optional loss.
        """
        x, y, validity_mask, cond_seq, prev_actions = (
            batch.x,
            batch.y,
            batch.validity_mask,
            batch.conditional_seq,
            batch.prev_actions,
        )
        x = self.encoder(x)

        if prev_actions is not None:
            prev_actions = self.act_dropout(prev_actions)
            x = torch.cat([x, prev_actions], dim=-1)

        B, T = x.shape[0:2]  # batch_size, sequence_length
        if T > self.context_len:
            raise ValueError(
                "The input sequence length exceeds the maximum context length."
            )

        x = self.transformer.fc_in(x)  # (batch_size, sequence_length, embed_dim)
        pos_idx = torch.arange(T).expand(B, -1).to(x.device)
        pos_embed = self.transformer.pos_embed(pos_idx)
        x += pos_embed
        if self.training and self.history_mask_prob > 0:
            validity_mask = self.mask_history_states(
                validity_mask, self.history_mask_prob
            )

        x = self.transformer.dropout(x)

        mask = self.mask[:T, :T]
        if self.conditional_seq_layer is not None and cond_seq is not None:
            T_ext = x.size(1) + cond_seq.size(1)
            _mask = torch.ones(T_ext, T_ext, dtype=torch.bool, device=x.device)
            _mask[-T:, -T:] = mask
            _mask[:-T, -T:] = False
            mask = _mask
            x_cond_seq = self.encoder(cond_seq)
            x_cond_seq = self.conditional_seq_layer(x_cond_seq)
            if self.transformer.goal_pos_embed is not None:
                cond_seq_len = x_cond_seq.size(1)
                goal_pos_idx = torch.arange(cond_seq_len, device=x_cond_seq.device)
                goal_pos_idx = goal_pos_idx.unsqueeze(0)
                goal_pos_emb = self.transformer.goal_pos_embed(goal_pos_idx)
                x_cond_seq = x_cond_seq + goal_pos_emb

            x = torch.cat([x_cond_seq, x], dim=1)
        for dec_block in self.transformer.dec_blocks:
            x = dec_block(
                x,
                attn_mask=~mask if (mask is not None) else None,
                key_padding_mask=~validity_mask
                if (validity_mask is not None and validity_mask.all())
                else None,
                **self.mha_kwargs,
            )
            x = x.nan_to_num()
        x = self.transformer.ln_f(x)
        gmm_params = self.get_gmm_params(
            x[:, -T:, ...]
        )  # -T: indexing is to account for conditioning tokens
        loss = None
        if y is not None:
            loss = self.compute_loss(
                gmm_params=gmm_params, y=y, validity_mask=validity_mask
            )

        return ModelOutput(output=gmm_params, loss=loss)

    @profile
    def compute_loss(
        self,
        gmm_params: BatchSequenceGMMParams,
        y: torch.Tensor,
        validity_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the neglog likelihood of the data under the model.

        Args:
            gmm_params (BatchedSequenceGMMParams): The predicted GMM parameters.
            y (torch.Tensor): The targets.
            validity_mask (torch.Tensor): A mask denoting the validity of the sequence.
                Defaults to None.

        Returns:
            torch.Tensor: The loss value.
        """
        distribution: MixtureSameFamily = self.get_distribution(gmm_params)
        neglog_prob = -distribution.log_prob(y)
        if validity_mask is not None:
            masked_loss = (neglog_prob * validity_mask).sum()
            loss = masked_loss / validity_mask.sum()
        else:
            loss = neglog_prob.mean()
        if self.lambda_mixtures != 0:
            loss += (
                self.lambda_mixtures
                * distribution.mixture_distribution.entropy().mean()
            )

        return loss

    def sampling_preprocessing(
        self,
        x: Union[torch.Tensor, NDArray],
        conditional_seq: Optional[Union[torch.Tensor, NDArray]] = None,
        prev_actions: Optional[Union[torch.Tensor, NDArray]] = None,
    ) -> Batch:
        """Preprocesses the input x for sampling, optionally handling prev_actions
        and conditional sequences (e.g. goals).

        Args:
            x (Union[torch.Tensor, NDArray]): The main input context (e.g., states).
            conditional_seq (Optional[Union[torch.Tensor, NDArray]]): Conditional sequence to be pre-appended to x.
            prev_actions (Optional[Union[torch.Tensor, NDArray]]): Previous actions that align with x.

        Returns:
            Batch: A dataclass with the preprocessed context x, optional prev_actions,
                optional conditional_seq, etc.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.ndim < 3:
            raise ValueError(
                "Expected the input x to have at least 3 dimensions (B, T, ...)."
            )
        T = x.shape[1]

        if conditional_seq is not None:
            if isinstance(conditional_seq, np.ndarray):
                conditional_seq = torch.from_numpy(conditional_seq)
            if conditional_seq.ndim < 3:
                raise ValueError(
                    "Expected conditional_seq to have at least 3 dimensions (B, T_cond, ...)."
                )

        if prev_actions is not None:
            if isinstance(prev_actions, np.ndarray):
                prev_actions = torch.from_numpy(prev_actions)
            if prev_actions.ndim < 3:
                raise ValueError(
                    "Expected prev_actions to have at least 3 dimensions (B, T, ...)."
                )
            if prev_actions.shape[1] != T:
                raise ValueError(
                    f"prev_actions' sequence length ({prev_actions.shape[1]}) "
                    f"must match x's sequence length ({T}) or be handled differently."
                )

        if self.pad_sampling and T < self.context_len:
            padding = torch.zeros(
                (x.shape[0], self.context_len - T, *x.shape[2:]),
                device=x.device,
                dtype=torch.float32,
            )
            x = torch.cat([padding, x], dim=1)

        if self.pad_sampling and prev_actions is not None:
            T_pa = prev_actions.shape[1]
            if T_pa < self.context_len:
                pa_padding = torch.zeros(
                    (
                        prev_actions.shape[0],
                        self.context_len - T_pa,
                        *prev_actions.shape[2:],
                    ),
                    device=prev_actions.device,
                    dtype=torch.float32,
                )
                prev_actions = torch.cat([pa_padding, prev_actions], dim=1)

        return Batch(
            x=x.to(self.device, dtype=torch.float32),
            prev_actions=(
                prev_actions.to(self.device, dtype=torch.float32)
                if prev_actions is not None
                else None
            ),
            conditional_seq=(
                conditional_seq.to(self.device, dtype=torch.float32)
                if conditional_seq is not None
                else None
            ),
        )

    def sample(self, **kwargs):
        if self.sample_fn == "gmm":
            return self.sample_gmm(**kwargs)
        elif self.sample_fn == "modes":
            return self.sample_modes(**kwargs)

    @torch.inference_mode()
    def sample_gmm(
        self,
        x: Union[torch.Tensor, NDArray],
        prev_actions: Optional[Union[torch.Tensor, NDArray]] = None,
        conditional_seq: Optional[Union[torch.Tensor, NDArray]] = None,
        return_output: bool = True,
        temperature: float = 1,
    ) -> Tuple[torch.Tensor, Optional[ModelOutput], Dict[str, int]]:
        """Samples the next token in the output sequence from a GMM predicted by the model.

        Args:
            x (Union[torch.Tensor, NDArray]): The input sequence or context.
            return_output (bool): Whether to return the model output.
            temperature (float): A scale to adjust the variance of the model.

        Returns:
            Tuple[torch.Tensor, Optional[ModelOutput]]: A tuple containing the sampled output,
            the model output (if return_output=True), and additional sampling details.
        """
        batch = self.sampling_preprocessing(
            x, conditional_seq=conditional_seq, prev_actions=prev_actions
        )
        out: ModelOutput = self.forward(batch)
        dist_params: BatchSequenceGMMParams = out.output

        if temperature != 1:
            dist_params = BatchSequenceGMMParams(
                locs=dist_params.locs,
                variances=dist_params.variances * temperature,
                mixture_probs=dist_params.mixture_probs,
            )

        dist = self.get_distribution(gmm_params=dist_params, last_only=True)
        sampled_k = dist.mixture_distribution.sample().item()
        sample = dist.component_distribution.sample()[:, sampled_k, :]
        return (
            sample,
            out if return_output else None,
            {
                "sampled_component": sampled_k,
                "mixture_probabilities": dist.mixture_distribution.probs.squeeze()
                .cpu()
                .numpy(),
                "distribution": dist,
            },
        )

    @torch.inference_mode()
    def sample_modes(
        self,
        x: Union[torch.Tensor, "NDArray"],
        prev_actions: Optional[Union[torch.Tensor, "NDArray"]] = None,
        conditional_seq: Optional[Union[torch.Tensor, "NDArray"]] = None,
        return_output: bool = True,
        max_iter: int = 1000,
        mode_tol: float = 1e-6,
        probs_tol: float = 1e-2,
        norm_tol: float = 1e-12,
        min_diff: float = 1e-8,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional["ModelOutput"], Dict[str, Any]]:
        """
        Example: Instead of returning just a single point from the set of modes,
        we build a mixture of local Gaussians around each valid mode, using
        the inverse Hessians as covariances.
        """

        def compute_grad_and_hessian_logp_vectorized(
            x_modes: torch.Tensor, distribution
        ):
            """
            Vectorized version of gradient and Hessian computation for log p(x)
            of a diagonal Gaussian mixture. Now returns the inverse of (-H_logp) at each x,
            but only for those points where -H_logp is positive-definite.

            Returns:
                max_eigvals: [M_valid]
                    The largest eigenvalue of the Hessian of log p(x)
                    for each valid mode (negative-definite H).
                logdet_hess_logp: [M_valid]
                    = log(det(-H_logp)), valid only if H is negative-definite.
                grad_norms: [M_valid]
                    Euclidean norm of ∇ log p(x).
                negH_inv: [M_valid, D, D]
                    The inverse of -H_logp for each valid mode x.
                valid_mask: [M]
                    Boolean mask indicating which input modes were valid
                    (i.e., H was negative-definite).
            """
            mixture_probs = distribution.mixture_distribution.probs
            means = distribution.component_distribution.loc
            variances = distribution.component_distribution.variances

            # --- Make sure shapes are consistent ---
            mixture_probs = mixture_probs.squeeze()
            if mixture_probs.ndim == 0:
                mixture_probs = mixture_probs[None]  # shape [K=1]

            means = means.squeeze()
            if means.ndim == 1:
                means = means.unsqueeze(0)  # shape [K, D]

            variances = variances.squeeze()
            if variances.ndim == 1:
                variances = variances.unsqueeze(0)  # shape [K, D]

            D = means.shape[1]
            # ----------------------------------------------------------------
            # 1) Compute per-component densities f_i(x) and mixture weights alpha_i(x)
            # ----------------------------------------------------------------
            x_expanded = x_modes.unsqueeze(1)  # [M, 1, D]
            means_expanded = means.unsqueeze(0)  # [1, K, D]
            diff = x_expanded - means_expanded  # [M, K, D]

            inv_var = 1.0 / variances.clamp(min=1e-12)  # [K, D]
            inv_var_expanded = inv_var.unsqueeze(0)  # [1, K, D]

            # log f_i(x): [M, K]
            log_fi = (
                -0.5 * (diff**2 * inv_var_expanded).sum(dim=-1)
                - 0.5 * D * math.log(2.0 * math.pi)
                - 0.5 * torch.log(variances).sum(dim=-1).unsqueeze(0)
            )
            pdf_values = log_fi.exp()  # [M, K]

            mixture_probs_expanded = mixture_probs.unsqueeze(0)  # [1, K]
            p_x = (mixture_probs_expanded * pdf_values).sum(dim=1)  # [M]
            alpha = (
                mixture_probs_expanded * pdf_values / p_x.unsqueeze(-1).clamp(min=1e-30)
            )  # [M, K]

            # ----------------------------------------------------------------
            # 2) grad and Hess log f_i(x)
            # ----------------------------------------------------------------
            grads_fi = -(diff * inv_var_expanded)  # [M, K, D]
            hess_fi = -inv_var_expanded  # [1, K, D] => broadcast => [M, K, D]

            # ----------------------------------------------------------------
            # 3) grad log p(x) = sum_i alpha_i * grad log f_i(x)
            # ----------------------------------------------------------------
            grad_logp = torch.sum(alpha.unsqueeze(-1) * grads_fi, dim=1)  # [M, D]
            grad_norms = grad_logp.norm(dim=-1)  # [M]

            # ----------------------------------------------------------------
            # 4) Hess log p(x) = sum_i alpha_i * Hess_fi
            #                    + sum_i [ grad_alpha_i ⊗ grad log f_i(x) ]
            # ----------------------------------------------------------------
            # termA: diagonal part from sum_i alpha_i * Hess_fi
            termA_diags = torch.sum(alpha.unsqueeze(-1) * hess_fi, dim=1)  # [M, D]
            termA = torch.diag_embed(termA_diags)  # [M, D, D]

            # grad_alpha_i = alpha_i * (grads_fi[i] - grad_logp)
            grad_alpha = alpha.unsqueeze(-1) * (
                grads_fi - grad_logp.unsqueeze(1)
            )  # [M, K, D]

            # termB: sum over i of (grad_alpha_i outer grads_fi[i])
            # shape => [M, K, D, D]
            grad_alpha_i_expanded = grad_alpha.unsqueeze(-1)  # [M, K, D, 1]
            grads_fi_i_expanded = grads_fi.unsqueeze(-2)  # [M, K, 1, D]
            termB_full = grad_alpha_i_expanded * grads_fi_i_expanded  # [M, K, D, D]
            termB = termB_full.sum(dim=1)  # [M, D, D]

            H_logp = termA + termB  # [M, D, D]   # Hessian of log p(x)

            # ----------------------------------------------------------------
            # 5) Check negative-definiteness: largest eigenvalue < 0
            #    Then invert -H if valid
            # ----------------------------------------------------------------
            eigvals, _ = torch.linalg.eigh(H_logp)  # [M, D], [M, D, D]
            max_eigvals = eigvals.max(dim=-1).values  # [M]

            # A local maximum => H is negative-definite => all eigenvalues < 0
            valid_mask = max_eigvals < 0

            # -H for invertible covariance (only for valid ones)
            negH = -H_logp[valid_mask]  # [M_valid, D, D]
            logdet_negH = torch.logdet(negH)  # [M_valid]
            negH_inv = torch.linalg.inv(negH)  # [M_valid, D, D]

            max_eigvals_valid = max_eigvals[valid_mask]
            logdet_hess_logp = logdet_negH
            grad_norms_valid = grad_norms[valid_mask]

            return (
                max_eigvals_valid,
                logdet_hess_logp,
                grad_norms_valid,
                negH_inv,
                valid_mask,
            )

        temp = kwargs.get("temperature", 1.0)

        with torch.inference_mode():
            batch = self.sampling_preprocessing(
                x, conditional_seq=conditional_seq, prev_actions=prev_actions
            )
            out: "ModelOutput" = self.forward(batch)
            dist_params: "BatchSequenceGMMParams" = out.output
            raw_dist_out = self.get_distribution(gmm_params=dist_params, last_only=True)

            probs = raw_dist_out.mixture_distribution.probs.squeeze()
            means = raw_dist_out.component_distribution.loc.squeeze()
            variances = raw_dist_out.component_distribution.variances.squeeze()

            valid_idx = (probs / probs.max()) > probs_tol
            probs = probs[valid_idx]
            probs = probs / probs.sum()
            means = means[valid_idx]
            variances = variances[valid_idx]

            dist_params = BatchSequenceGMMParams(
                variances=variances[None, None, ...],
                locs=means[None, None, ...],
                mixture_probs=probs[None, None, ...],
            )
            dist_out = self.get_distribution(gmm_params=dist_params, last_only=True)

            # If there's only 1 component, its single mode is just the mean
            K, dim = means.shape
            if K == 1:
                single_mode = (
                    means.clone()
                    + MultivariateNormalDiag(
                        loc=torch.zeros_like(means), variances=temp * variances
                    ).sample()
                )
                return (
                    single_mode.unsqueeze(0).unsqueeze(0),
                    out if return_output else None,
                    {
                        "sampled_component": 0,
                        "mixture_probabilities": probs.detach().cpu().numpy(),
                        "distribution": raw_dist_out,
                        "valid_modes": single_mode,
                    },
                )

            init_list = []
            for _ in range(K):
                w = torch.rand(K, device=means.device)
                w = w / w.sum()
                init_pt = (w[:, None] * means).sum(dim=0)
                init_list.append(init_pt)
            init_list += [m for m in means]
            modes = torch.stack(init_list, dim=0)

        converged = torch.zeros(modes.shape[0], dtype=torch.bool, device=modes.device)
        base_normalization = math.sqrt((2 * math.pi) ** dim)
        log_var = variances.log().sum(dim=-1)

        for it in range(max_iter):
            not_converged = ~converged
            if not torch.any(not_converged):
                logger.debug(f"All modes converged at iteration {it}.")
                break

            diff = modes[not_converged, None, :] - means[None, :, :]  # [Nc, K, D]
            dist = torch.sum(diff**2 / variances[None, ...], dim=-1)  # [Nc, K]
            normalization = base_normalization * torch.exp(0.5 * log_var)  # [K]
            normalization = torch.clamp(normalization, min=norm_tol)
            likelihoods = torch.exp(-0.5 * dist) / normalization[None, :]  # [Nc, K]

            weighted_means = (probs[None, :] * likelihoods)[:, :, None] * means[
                None, :, :
            ]
            numerator = (weighted_means / variances[None, :, :]).sum(dim=1)  # [Nc, D]
            denominator = (
                (probs[None, :] * likelihoods)[..., None] / variances[None, ...]
            ).sum(dim=1)  # [Nc, D]
            denominator = torch.clamp(denominator, min=norm_tol)
            updated_modes = numerator / denominator  # [Nc, D]

            movement = (updated_modes - modes[not_converged]).norm(dim=-1)
            modes[not_converged] = updated_modes
            newly_converged = movement < mode_tol
            converged[not_converged] |= newly_converged

        mode_loglikelihoods = dist_out.log_prob(modes)  # [num_init]
        _, logdet_hess, _, negH_inv, valid_mask = (
            compute_grad_and_hessian_logp_vectorized(modes, dist_out)
        )

        if not valid_mask.any():
            logger.warning(
                "No valid negative-definite Hessians found. Fallback sampler."
            )
            return self.sample_gmm(x=x, **kwargs)
        modes = modes[valid_mask]
        mode_loglikelihoods = mode_loglikelihoods[valid_mask]

        def merge_close_modes(modes_, mode_lls, min_diff):
            sorted_indices = torch.argsort(mode_lls, descending=True)
            selected_indices = []
            for i in sorted_indices:
                candidate = modes_[i]
                too_close = any(
                    (torch.norm(candidate - modes_[sel_idx]) ** 2 < min_diff)
                    for sel_idx in selected_indices
                )
                if not too_close:
                    selected_indices.append(i)
            return torch.stack(selected_indices)

        selected_indices = merge_close_modes(
            modes, mode_loglikelihoods, min_diff=min_diff
        )
        modes = modes[selected_indices]  # shape [Nm, D]
        mode_loglikelihoods = mode_loglikelihoods[selected_indices]
        logdet_hess_logp = logdet_hess[selected_indices]
        negH_inv = negH_inv[selected_indices]  # shape [Nm, D, D]

        weights = mode_loglikelihoods - 0.5 * logdet_hess_logp
        mode_dist = torch.distributions.Categorical(logits=weights)
        mvn = torch.distributions.MultivariateNormal(
            loc=modes,  # shape [Nm, D]
            covariance_matrix=negH_inv * temp,  # shape [Nm, D, D]
        )
        mix_dist = torch.distributions.MixtureSameFamily(
            mode_dist,
            mvn,
        )
        sample = mix_dist.sample()
        sample = sample.unsqueeze(0).unsqueeze(0)
        return (
            sample.detach(),
            out if return_output else None,
            {
                "sampled_component": None,
                "mixture_probabilities": probs.detach().cpu().numpy(),
                "distribution": raw_dist_out,
                "valid_modes": modes,
            },
        )

    def sample_from_model_output(
        self,
        x: ModelOutput,
        temperature: Optional[float] = None,
        last_only: bool = False,
    ) -> torch.Tensor:
        """Samples from the model's output"""
        dist_params: BatchSequenceGMMParams = x.output
        if last_only:
            dist_params = dist_params[:, -1, ...]
        if temperature is not None or temperature != 1:
            dist_params = BatchSequenceGMMParams(
                locs=dist_params.locs,
                variances=dist_params.variances * temperature,
                mixture_probs=dist_params.mixture_probs,
            )
        dist = self.get_distribution(gmm_params=x.output, last_only=True)
        return dist.sample()
