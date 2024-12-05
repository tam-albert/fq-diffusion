import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward

from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.function_utils import get_decompose_dim, get_init_scale
from flatquant.trans_utils import InvDecomposeTransMatrix, SVDDecomposeTransMatrix
from flatquant.utils import skip_initialization

logger = logging.getLogger(__name__)


class FlatQuantPixArtFeedForward(nn.Module):
    def __init__(self, args, module: FeedForward):
        super().__init__()
        self.args = args

        # i actually dgaf
        up_proj = module.net[0].proj
        # there is a gelu (tanh approximation) in the middle
        down_proj = module.net[-1]

        self.up_proj = FlatQuantizedLinear(args, up_proj)
        self.down_proj = FlatQuantizedLinear(args, down_proj)
        self.add_fq_trans()

        self._ori_mode = False
        self.diag_init = args.diag_init
        if self.diag_init == "sq_style":
            self.up_smax = (
                torch.ones_like(self.up_proj.linear.weight.abs().max(dim=0)[0]).cuda()
                * 1e-5
            )
            self.down_smax = (
                torch.ones_like(self.down_proj.linear.weight.abs().max(dim=0)[0]).cuda()
                * 1e-5
            )

    def act_fn(self, x):
        return F.gelu(x, approximate="tanh")

    def add_fq_trans(self):
        if self.args.direct_inv:
            DecomposeTransMatrix = InvDecomposeTransMatrix
        else:
            DecomposeTransMatrix = SVDDecomposeTransMatrix
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            up_dim_left, up_dim_right = get_decompose_dim(
                self.up_proj.linear.weight.shape[1]
            )
            self.up_trans = DecomposeTransMatrix(
                up_dim_left, up_dim_right, add_diag=self.args.add_diag
            )
            down_dim_left, down_dim_right = get_decompose_dim(
                self.down_proj.linear.weight.shape[1]
            )
            self.down_trans = DecomposeTransMatrix(
                down_dim_left, down_dim_right, add_diag=self.args.add_diag
            )
        else:
            self.up_trans, self.down_trans = None, None

    def _trans_forward(self, x):
        if self.up_trans is not None:
            x_ts = self.up_trans(x)
        else:
            x_ts = x
        up_states = self.up_proj(x_ts, qa_trans=self.up_trans)

        x_act_fn = self.act_fn(up_states)
        if self.down_trans is not None:
            x_ts_2 = self.down_trans(x_act_fn)
        else:
            x_ts_2 = x_act_fn
        down_states = self.down_proj(x_ts_2, qa_trans=self.down_trans)
        return down_states

    def _ori_forward(self, x):
        """origin implement: down_proj = self.down_proj(self.act_fn(self.up_proj(x))"""
        if self.diag_init == "sq_style":
            self.up_smax = torch.maximum(
                self.up_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        x = self.act_fn(self.up_proj._ori_forward(x))
        if self.diag_init == "sq_style":
            self.down_smax = torch.maximum(
                self.down_smax,
                x.reshape(-1, x.shape[-1]).abs().max(0)[0].clone().detach(),
            )
        down_states = self.down_proj._ori_forward(x)
        return down_states

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)

    def reparameterize(
        self,
    ):
        if self.up_trans is not None:
            self.up_trans.to_eval_mode()
            self.down_trans.to_eval_mode()
        self.up_proj.reparameterize(qa_trans=self.up_trans)
        self.down_proj.reparameterize(qa_trans=self.down_trans)
        if self.up_trans is not None:
            self.up_trans.use_diag = False
        # merge trans's diag scale
        if self.down_trans is not None and self.down_trans.add_diag:
            up_weight = self.up_proj.linear.weight
            ori_dtype = up_weight.dtype
            up_weight = (
                up_weight.to(torch.float64)
                .T.mul(self.down_trans.diag_scale.to(torch.float64))
                .T
            )
            self.up_proj.linear.weight.data = up_weight.to(ori_dtype)
            self.down_trans.use_diag = False

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "up_smax") and hasattr(self, "down_smax")
        upw_smax = self.up_proj.linear.weight.abs().max(dim=0)[0]
        downw_smax = self.down_proj.linear.weight.abs().max(dim=0)[0]
        if self.up_trans is not None:
            self.up_trans.diag_scale.data = get_init_scale(
                upw_smax, self.up_smax, alpha
            )
        if self.down_trans is not None:
            self.down_trans.diag_scale.data = get_init_scale(
                downw_smax, self.down_smax, alpha
            )
        del self.up_smax, self.down_smax
        self.diag_init = None

    def rep_matrix_only(
        self,
    ):
        if self.up_trans is not None:
            self.up_trans.to_eval_mode()
            self.down_trans.to_eval_mode()


def apply_flatquant_to_pixart(args, model):
    skip_initialization()
    for layer_i in range(model.config.num_layers):
        # feedforward
        model.transformer_blocks[layer_i].ff = FlatQuantPixArtFeedForward(
            args, model.transformer_blocks[layer_i].ff
        )
    return model
