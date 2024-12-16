import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import Attention

from flatquant.flat_linear import FlatQuantizedLinear
from flatquant.function_utils import get_decompose_dim, get_init_scale
from flatquant.trans_utils import InvDecomposeTransMatrix, SVDDecomposeTransMatrix
from flatquant.trans_utils import InvSingleTransMatrix, SVDSingleTransMatrix

from flatquant.utils import skip_initialization
from flatquant.quant_utils import ActivationQuantizer


class FlatQuantPixArtFeedForward(nn.Module):
    def __init__(self, args, module: FeedForward):
        self._ori_mode = True
        super().__init__()
        self.args = args

        # i actually dgaf
        up_proj = module.net[0].proj
        # there is a gelu (tanh approximation) in the middle
        down_proj = module.net[-1]

        self.old_net = module

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
        #return self.test_out_proj(self.test_dropout(self.test_in_proj(x)))
        #return self.old_net(x)
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
        # print(f"here, with ori mode as {self._ori_mode}")
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


class FlatQuantPixArtAttentionBeta(nn.Module):
    def __init__(self, args, module: Attention, self_attention = False):
        self._ori_mode = True
        super().__init__()
        self.args = args

        # set kv precision. allows for different
        # precision for self and cross attention (as suggested by TAs)
        self.self_attention = self_attention
        self.args.k_bits = self.args.k_cross_bits if (not self_attention and args.k_cross_bits > 0) else self.args.k_bits
        self.args.v_bits = self.args.v_cross_bits if (not self_attention and args.v_cross_bits > 0) else self.args.v_bits

        # hard coding hidden sizes and attention heads. probably an easier way to do this...
        self.num_attention_heads = module.heads
        # copy over jsut for ease
        self.heads = module.heads
        self.head_dim = module.inner_dim // module.heads
        self.residual_connection = module.residual_connection
        self.rescale_output_factor = module.rescale_output_factor

        self.to_q = FlatQuantizedLinear(args, module.to_q)
        self.to_k = FlatQuantizedLinear(args, module.to_k)
        self.to_v = FlatQuantizedLinear(args, module.to_v)
        self.to_out = FlatQuantizedLinear(args, module.to_out[0])
        self.add_fq_trans()

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(
                bits=args.q_bits,
                sym=not (args.q_asym),
                lac=args.lac,
                groupsize=-1,
            )
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(
                bits=args.k_bits,
                sym=not (args.k_asym),
                lac=args.lac,
                groupsize=-1,
            )
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(
                bits=args.v_bits,
                sym=not (args.v_asym),
                lac=args.lac,
                groupsize=-1,
            )

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init

        if self.diag_init == "sq_style":
            self.ln_smax = (
                torch.ones_like(self.to_q.linear.weight.abs().max(dim=0)[0]).cuda()
                * 1e-5
            )

    def add_fq_trans(self):
        # self._ori_mode = True
        if self.args.direct_inv:
            SingleTransMatrix, DecomposeTransMatrix = (
                InvSingleTransMatrix,
                InvDecomposeTransMatrix,
            )
        else:
            SingleTransMatrix, DecomposeTransMatrix = (
                SVDSingleTransMatrix,
                SVDDecomposeTransMatrix,
            )
        
        # TODO: explore using diff rotation mats for cross attention?
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(
                self.to_q.linear.weight.shape[1]
            )
            self.ln_trans = DecomposeTransMatrix(
                ln_dim_left, ln_dim_right, add_diag=self.args.add_diag
            )
            self.o_trans = SingleTransMatrix(self.num_attention_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kc_trans = SingleTransMatrix(self.head_dim)
        else:
            self.kc_trans = None

        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vc_trans = SingleTransMatrix(self.head_dim)
        else:
            self.vc_trans = None

    def _trans_forward_after_ln(self, hidden_states, encoder_hidden_states):
        # self._ori_mode = True
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
            if encoder_hidden_states is not None:
                encoder_hidden_states = self.ln_trans(encoder_hidden_states)
            
        # self attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # do we want to use a diff rotation for the k, v matrices 
        # in cross attention? or is it fine....
        query_states = self.to_q(hidden_states, qa_trans=self.ln_trans)
        key_states = self.to_k(encoder_hidden_states, qa_trans=self.ln_trans)

        if self.args.separate_vtrans:
            value_states = self.to_v(encoder_hidden_states, qa_trans=self.ln_trans)
        else:
            value_states = self.to_v(encoder_hidden_states, qa_trans=self.ln_trans, out_trans=self.vc_trans)
        
        return query_states, key_states, value_states
    
    def _ori_forward_after_ln(self, hidden_states, encoder_hidden_states):
        # self._ori_mode = True
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            self.ln_smax = torch.maximum(
                self.ln_smax,
                hidden_states.reshape(-1, hidden_states.shape[-1])
                .abs()
                .max(0)[0]
                .clone()
                .detach()
            )
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query_states = self.to_q._ori_forward(hidden_states)
        key_states = self.to_k._ori_forward(encoder_hidden_states)
        value_states = self.to_v._ori_forward(encoder_hidden_states)

        return query_states, key_states, value_states
    
    def quant_v(self, value_states):
        # self._ori_mode = True
        if self.args.separate_vtrans:
            value_states = self.vc_trans(value_states)
        if self.args.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states
    
    def quant_k(self, q, k):
        # self._ori_mode = True
        if not (self.args.k_bits < 16 or self.args.q_bits < 16):
            return q, k
        # Q/K transform
        if self.kc_trans is not None:
            q = self.kc_trans(q, inv_t=True)
            k = self.kc_trans(k)
        if self.args.q_bits < 16:
            q = self.q_cache_quantizer(q).to(q)
        # TODO: by default do the per-head quantizaion for k-v-cache
        if self.args.k_bits < 16:
            k = self.k_cache_quantizer(k).to(q)
        return q, k
    
    def forward(
        self,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
    ):  
        # self._ori_mode = True
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(
                hidden_states,
                encoder_hidden_states
            )
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(
                hidden_states,
                encoder_hidden_states
            )
        
        if attention_mask is not None:
            """
            print(attention_mask)
            print(f"attention mask shape pre prep is {attention_mask.shape}")
            print(f"query shapes: {query_states.shape}")
            print(f"key shapes: {key_states.shape}")
            print(f"value shapes: {value_states.shape}")
            """
            attention_mask = Attention.prepare_attention_mask(self, attention_mask,sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.num_attention_heads, -1, attention_mask.shape[-1])
            """
            print(f"attention mask shape post prep is {attention_mask.shape}")
            """
                
        inner_dim = key_states.shape[-1]
        head_dim = self.head_dim

        query_states = query_states.view(batch_size, -1, self.num_attention_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_attention_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_attention_heads, head_dim).transpose(1, 2)

        if not self._ori_mode:
            query_states, key_states = self.quant_k(query_states, key_states)
            value_states = self.quant_v(value_states)
        # hidden states = F.scaled_dot_product_attention(query, key, value, attn_mask = attention_mask)
        # ugly way of doing it

        #print(f"query shapes: {query_states.shape}")
        #print(f"key shapes: {key_states.shape}")
        #print(f"value shapes: {value_states.shape}")
        # where the fuck is dividing by sqrt(k)
        """
        attention_weights = query_states @ key_states.transpose(-2, -1)
        if attention_mask is not None:
            attention_weights += attention_mask
        # upcast attention to fp32
        print(attention_weights.shape, attention_weights.dtype)
        current = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory after attention matrix materialized before softmaxing: {current:.2f}MB")
        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # print(f"attention weights shape {attention_weights.shape}")
        hidden_states = attention_weights @ value_states
        """
        
        hidden_states = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_attention_heads * head_dim)

        if self._ori_mode:
            hidden_states = self.to_out._ori_forward(hidden_states)
        else:
            if self.o_trans is None and self.vc_trans is not None:
                init_shape = hidden_states.shape
                hidden_states = hidden_states.reshape(
                    -1,
                    self.num_attention_heads,
                    self.head_dim
                )
                hidden_states = torch.matmul(
                    hidden_states,
                    self.vc_trans.get_matrix(inv_t=True).T.to(hidden_states)
                ).reshape(init_shape)
                hidden_states = self.to_out(hidden_states)
            else:
                init_shape = hidden_states.shape
                hidden_states = hidden_states.reshape(
                    -1,
                    self.num_attention_heads,
                    self.head_dim
                )
                hidden_states = torch.matmul(
                    self.o_trans.get_matrix().T.to(hidden_states), hidden_states
                ).reshape(init_shape)
                if not self._eval_mode:
                    attn_o_og_it = self.o_trans.get_matrix(inv_t = True)
                    attn_v_og_it = self.vc_trans.get_matrix(inv_t = True)
                    hidden_states = self.to_out(
                        hidden_states, qa_trans=[attn_o_og_it, attn_v_og_it]
                    )
                else:
                    hidden_states = self.to_out(hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        hidden_states = hidden_states / self.rescale_output_factor
        
        return hidden_states
    
    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kc_trans is not None:
            self.kc_trans.to_eval_mode()
        if self.vc_trans is not None:
            self.vc_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.to_q.reparameterize(qa_trans = self.ln_trans)
        self.to_k.reparameterize(qa_trans = self.ln_trans)
        if self.args.separate_vtrans:
            self.to_v.reparameterize(qa_trans=self.ln_trans)
        else:
            self.to_v.reparameterize(
                qa_trans=self.ln_trans, out_trans=self.vc_trans
            )
        if self.o_trans is not None and self.vc_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vc_trans.get_matrix(inv_t=True)
            self.to_out.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "ln_smax")
        qkvw_smax = (
            torch.cat(
                [
                    self.to_q.linear.weight,
                    self.to_k.linear.weight,
                    self.to_v.linear.weight,
                ],
                dim=0,
            )
            .abs()
            .max(dim=0)[0]
        )
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(
                qkvw_smax, self.ln_smax, alpha
            )
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(
        self,
    ):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kc_trans is not None:
            self.kc_trans.to_eval_mode()
        if self.vc_trans is not None:
            self.vc_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()       

class FlatQuantPixArtAttention(nn.Module):
    def __init__(self, args, module: Attention, self_attention = False):
        self._ori_mode = True
        super().__init__()
        self.args = args

        # set kv precision. allows for different
        # precision for self and cross attention (as suggested by TAs)
        self.self_attention = self_attention
        self.args.k_bits = self.args.k_cross_bits if (not self_attention and args.k_cross_bits > 0) else self.args.k_bits
        self.args.v_bits = self.args.v_cross_bits if (not self_attention and args.v_cross_bits > 0) else self.args.v_bits

        # hard coding hidden sizes and attention heads. probably an easier way to do this...
        self.num_attention_heads = module.heads
        # copy over jsut for ease
        self.heads = module.heads
        self.head_dim = module.inner_dim // module.heads
        self.residual_connection = module.residual_connection
        self.rescale_output_factor = module.rescale_output_factor

        self.to_q = FlatQuantizedLinear(args, module.to_q)
        self.to_k = FlatQuantizedLinear(args, module.to_k)
        self.to_v = FlatQuantizedLinear(args, module.to_v)
        self.to_out = FlatQuantizedLinear(args, module.to_out[0])
        self.add_fq_trans()

        if args.q_bits < 16:
            self.q_cache_quantizer = ActivationQuantizer(
                bits=args.q_bits,
                sym=not (args.q_asym),
                lac=args.lac,
                groupsize=-1,
            )
        if args.k_bits < 16:
            self.k_cache_quantizer = ActivationQuantizer(
                bits=args.k_bits,
                sym=not (args.k_asym),
                lac=args.lac,
                groupsize=-1,
            )
        if args.v_bits < 16:
            self.v_cache_quantizer = ActivationQuantizer(
                bits=args.v_bits,
                sym=not (args.v_asym),
                lac=args.lac,
                groupsize=-1,
            )

        self._ori_mode = False
        self._eval_mode = False
        self.diag_init = args.diag_init

        if self.diag_init == "sq_style":
            self.ln_smax = (
                torch.ones_like(self.to_q.linear.weight.abs().max(dim=0)[0]).cuda()
                * 1e-5
            )

    def add_fq_trans(self):
        # self._ori_mode = True
        if self.args.direct_inv:
            SingleTransMatrix, DecomposeTransMatrix = (
                InvSingleTransMatrix,
                InvDecomposeTransMatrix,
            )
        else:
            SingleTransMatrix, DecomposeTransMatrix = (
                SVDSingleTransMatrix,
                SVDDecomposeTransMatrix,
            )
        
        # TODO: explore using diff rotation mats for cross attention?
        if self.args.w_bits < 16 or self.args.a_bits < 16:
            ln_dim_left, ln_dim_right = get_decompose_dim(
                self.to_q.linear.weight.shape[1]
            )
            self.ln_trans = DecomposeTransMatrix(
                ln_dim_left, ln_dim_right, add_diag=self.args.add_diag
            )
            if not self.self_attention:
                ln_external_left, ln_external_right = get_decompose_dim(
                    self.to_k.linear.weight.shape[1]
                )
                self.external_ln_trans = DecomposeTransMatrix(
                    ln_external_left, ln_external_right, add_diag=self.args.add_diag
                )
            self.o_trans = SingleTransMatrix(self.num_attention_heads)
        else:
            self.ln_trans, self.o_trans = None, None

        if self.args.k_bits < 16 or self.args.q_bits < 16:
            self.kc_trans = SingleTransMatrix(self.head_dim)
        else:
            self.kc_trans = None

        if self.args.v_bits < 16 or self.args.w_bits < 16 or self.args.a_bits < 16:
            self.vc_trans = SingleTransMatrix(self.head_dim)
        else:
            self.vc_trans = None

    def _trans_forward_after_ln(self, hidden_states, encoder_hidden_states):
        # self._ori_mode = True
        if self.ln_trans is not None:
            hidden_states = self.ln_trans(hidden_states)
            if encoder_hidden_states is not None:
                encoder_hidden_states = self.external_ln_trans(encoder_hidden_states)
            
        # self attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # do we want to use a diff rotation for the k, v matrices 
        # in cross attention? or is it fine....
        query_states = self.to_q(hidden_states, qa_trans=self.ln_trans)
        if self.self_attention:
            key_states = self.to_k(encoder_hidden_states, qa_trans=self.ln_trans)
        else:
            key_states = self.to_k(encoder_hidden_states, qa_trans=self.external_ln_trans)

        if self.args.separate_vtrans:
            if self.self_attention:
                value_states = self.to_v(encoder_hidden_states, qa_trans=self.ln_trans)
            else:
                value_states = self.to_v(encoder_hidden_states, qa_trans=self.external_ln_trans)
        else:
            if self.self_attention:
                value_states = self.to_v(encoder_hidden_states, qa_trans=self.ln_trans, out_trans=self.vc_trans)
            else:
                value_states = self.to_v(encoder_hidden_states, qa_trans=self.external_ln_trans, out_trans=self.vc_trans)
        
        return query_states, key_states, value_states
    
    def _ori_forward_after_ln(self, hidden_states, encoder_hidden_states):
        # self._ori_mode = True
        if self.diag_init == "sq_style" and hasattr(self, "ln_smax"):
            self.ln_smax = torch.maximum(
                self.ln_smax,
                hidden_states.reshape(-1, hidden_states.shape[-1])
                .abs()
                .max(0)[0]
                .clone()
                .detach()
            )
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query_states = self.to_q._ori_forward(hidden_states)
        key_states = self.to_k._ori_forward(encoder_hidden_states)
        value_states = self.to_v._ori_forward(encoder_hidden_states)

        return query_states, key_states, value_states
    
    def quant_v(self, value_states):
        # self._ori_mode = True
        if self.args.separate_vtrans:
            value_states = self.vc_trans(value_states)
        if self.args.v_bits < 16:
            value_states = self.v_cache_quantizer(value_states)
        return value_states
    
    def quant_k(self, q, k):
        # self._ori_mode = True
        if not (self.args.k_bits < 16 or self.args.q_bits < 16):
            return q, k
        # Q/K transform
        if self.kc_trans is not None:
            q = self.kc_trans(q, inv_t=True)
            k = self.kc_trans(k)
        if self.args.q_bits < 16:
            q = self.q_cache_quantizer(q).to(q)
        # TODO: by default do the per-head quantizaion for k-v-cache
        if self.args.k_bits < 16:
            k = self.k_cache_quantizer(k).to(q)
        return q, k
    
    def forward(
        self,
        hidden_states,
        encoder_hidden_states = None,
        attention_mask = None,
    ):  
        # self._ori_mode = True
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if self._ori_mode:
            query_states, key_states, value_states = self._ori_forward_after_ln(
                hidden_states,
                encoder_hidden_states
            )
        else:
            query_states, key_states, value_states = self._trans_forward_after_ln(
                hidden_states,
                encoder_hidden_states
            )
        
        if attention_mask is not None:
            """
            print(attention_mask)
            print(f"attention mask shape pre prep is {attention_mask.shape}")
            print(f"query shapes: {query_states.shape}")
            print(f"key shapes: {key_states.shape}")
            print(f"value shapes: {value_states.shape}")
            """
            attention_mask = Attention.prepare_attention_mask(self, attention_mask,sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.num_attention_heads, -1, attention_mask.shape[-1])
            """
            print(f"attention mask shape post prep is {attention_mask.shape}")
            """
                
        inner_dim = key_states.shape[-1]
        head_dim = self.head_dim

        query_states = query_states.view(batch_size, -1, self.num_attention_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_attention_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_attention_heads, head_dim).transpose(1, 2)

        if not self._ori_mode:
            query_states, key_states = self.quant_k(query_states, key_states)
            value_states = self.quant_v(value_states)
        # hidden states = F.scaled_dot_product_attention(query, key, value, attn_mask = attention_mask)
        # ugly way of doing it

        #print(f"query shapes: {query_states.shape}")
        #print(f"key shapes: {key_states.shape}")
        #print(f"value shapes: {value_states.shape}")
        # where the fuck is dividing by sqrt(k)
        """
        attention_weights = query_states @ key_states.transpose(-2, -1)
        if attention_mask is not None:
            attention_weights += attention_mask
        # upcast attention to fp32
        print(attention_weights.shape, attention_weights.dtype)
        current = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory after attention matrix materialized before softmaxing: {current:.2f}MB")
        attention_weights = nn.functional.softmax(attention_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # print(f"attention weights shape {attention_weights.shape}")
        hidden_states = attention_weights @ value_states
        """
        
        hidden_states = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.num_attention_heads * head_dim)

        if self._ori_mode:
            hidden_states = self.to_out._ori_forward(hidden_states)
        else:
            if self.o_trans is None and self.vc_trans is not None:
                init_shape = hidden_states.shape
                hidden_states = hidden_states.reshape(
                    -1,
                    self.num_attention_heads,
                    self.head_dim
                )
                hidden_states = torch.matmul(
                    hidden_states,
                    self.vc_trans.get_matrix(inv_t=True).T.to(hidden_states)
                ).reshape(init_shape)
                hidden_states = self.to_out(hidden_states)
            else:
                init_shape = hidden_states.shape
                hidden_states = hidden_states.reshape(
                    -1,
                    self.num_attention_heads,
                    self.head_dim
                )
                hidden_states = torch.matmul(
                    self.o_trans.get_matrix().T.to(hidden_states), hidden_states
                ).reshape(init_shape)
                if not self._eval_mode:
                    attn_o_og_it = self.o_trans.get_matrix(inv_t = True)
                    attn_v_og_it = self.vc_trans.get_matrix(inv_t = True)
                    hidden_states = self.to_out(
                        hidden_states, qa_trans=[attn_o_og_it, attn_v_og_it]
                    )
                else:
                    hidden_states = self.to_out(hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        hidden_states = hidden_states / self.rescale_output_factor
        
        return hidden_states
    
    def reparameterize(self):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kc_trans is not None:
            self.kc_trans.to_eval_mode()
        if self.vc_trans is not None:
            self.vc_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()
        self.to_q.reparameterize(qa_trans = self.ln_trans)
        self.to_k.reparameterize(qa_trans = self.ln_trans)
        if self.args.separate_vtrans:
            self.to_v.reparameterize(qa_trans=self.ln_trans)
        else:
            self.to_v.reparameterize(
                qa_trans=self.ln_trans, out_trans=self.vc_trans
            )
        if self.o_trans is not None and self.vc_trans is not None:
            attn_o_og_it = self.o_trans.get_matrix(inv_t=True)
            attn_v_og_it = self.vc_trans.get_matrix(inv_t=True)
            self.to_out.reparameterize(qa_trans=[attn_o_og_it, attn_v_og_it])
        self._eval_mode = True

    def init_diag_scale(self, alpha=0.5):
        assert hasattr(self, "ln_smax")
        qkvw_smax = (
            torch.cat(
                [
                    self.to_q.linear.weight,
                    self.to_k.linear.weight,
                    self.to_v.linear.weight,
                ],
                dim=0,
            )
            .abs()
            .max(dim=0)[0]
        )
        if self.ln_trans is not None:
            self.ln_trans.diag_scale.data = get_init_scale(
                qkvw_smax, self.ln_smax, alpha
            )
        if not self.self_attention:
            self.external_ln_trans.diag_scale.data = get_init_scale(
                qkvw_smax, self.ln_smax, alpha
            )
        del self.ln_smax
        self.diag_init = None

    def rep_matrix_only(
        self,
    ):
        if self.ln_trans is not None:
            self.ln_trans.to_eval_mode()
        if self.kc_trans is not None:
            self.kc_trans.to_eval_mode()
        if self.vc_trans is not None:
            self.vc_trans.to_eval_mode()
        if self.o_trans is not None:
            self.o_trans.to_eval_mode()    

def apply_flatquant_to_pixart(args, model, shared = True):
    skip_initialization()
    for layer_i in range(model.config.num_layers):
        # feedforward
        
        model.transformer_blocks[layer_i].ff = FlatQuantPixArtFeedForward(
            args, model.transformer_blocks[layer_i].ff
        )
        if shared:
            print("here!")
            model.transformer_blocks[layer_i].attn1 = FlatQuantPixArtAttentionBeta(
                args, model.transformer_blocks[layer_i].attn1, self_attention = True
            )
            model.transformer_blocks[layer_i].attn2 = FlatQuantPixArtAttentionBeta(
                args, model.transformer_blocks[layer_i].attn2, self_attention = False
            )
        else:
            model.transformer_blocks[layer_i].attn1 = FlatQuantPixArtAttention(
                args, model.transformer_blocks[layer_i].attn1, self_attention = True
            )
            model.transformer_blocks[layer_i].attn2 = FlatQuantPixArtAttention(
                args, model.transformer_blocks[layer_i].attn2, self_attention = False
            )   
        
    return model
