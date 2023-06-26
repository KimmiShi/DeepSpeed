# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from deepspeed.utils import log_dist

from deepspeed.utils import groups
from .sharded_moe import MOELayer, TopKGate
from .experts import Experts
import typing


class MoE(torch.nn.Module):
    """Initialize an MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        use_residual (bool, optional): default=False, make this MoE layer a Residual MoE (https://arxiv.org/abs/2201.05596) layer.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        use_tutel (bool, optional): default=False, whether to use Tutel optimizations (if installed).
        enable_expert_tensor_parallelism (bool, optional): default=False, whether to use tensor parallelism for experts
    """

    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 enable_expert_tensor_parallelism: bool = False):

        super(MoE, self).__init__()

        self.use_residual = use_residual
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

        log_dist(
            f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}',
            [0])

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        self.deepspeed_moe = MOELayer(TopKGate(hidden_size, num_experts, k, capacity_factor, eval_capacity_factor,
                                               min_capacity, noisy_gate_policy, drop_tokens, use_rts),
                                      experts,
                                      self.expert_group_name,
                                      self.ep_size,
                                      self.num_local_experts,
                                      use_tutel=use_tutel)
        if self.use_residual:
            self.mlp = expert
            # coefficient is used for weighted sum of the output of expert and mlp
            self.coefficient = torch.nn.Linear(hidden_size, 2)

    def set_deepspeed_parallelism(self):
        self._create_process_groups()

    def _create_process_groups(self):
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(self.ep_size)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(self.ep_size, mpu=groups.mpu)
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.deepspeed_moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if type(output_mlp) is tuple:
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts

import fmoe
from fmoe import FMoE
from fmoe.linear import FMoELinear
import copy

class FMoEMLPLinear(FMoELinear):
    def __init__(self, num_expert: int, fc:torch.nn.Module, rank: int = 0):
        in_feat = fc.weight.shape[1]
        out_feat = fc.weight.shape[0]
        bias=hasattr(fc, 'bias')
        super().__init__(num_expert, in_feat, out_feat, bias, rank)

        self.weight.data.copy_(fc.weight.data)
        if bias:
            self.bias.data.copy_(fc.bias.data)

class MlpExpert(torch.nn.Module):
    def __init__(self, mlp, ep_rank=0, num_expert=1):
        super().__init__()
        # self.fc1 = FMoEMLPLinear(num_expert, mlp.fc1, rank=ep_rank)
        # self.fc2 = FMoEMLPLinear(num_expert, mlp.fc2, rank=ep_rank)
        self.fc1 = copy.deepcopy(mlp.fc1)
        self.fc2 = copy.deepcopy(mlp.fc2)

        self.act = copy.deepcopy(mlp.act)
        self.drop1 = copy.deepcopy(mlp.drop1)
        self.drop2 = copy.deepcopy(mlp.drop2)

    def forward(self, x, cnt):
        # print("rank: ", torch.distributed.get_rank(), "cnt:", cnt)
        # x = self.fc1(x, cnt)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        # x = self.fc2(x, cnt)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class VitFMoE(FMoE):
    def __init__(
        self, experts, num_expert=1, d_model=1, top_k=1, moe_group=None, expert_kwargs={}
    ):
        world_size = torch.distributed.get_world_size(moe_group)
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            world_size=world_size,
            mp_group=None,
            top_k=top_k,
            moe_group=moe_group,
            # gate = fmoe.gates.GShardGate,
        )
        # expert_kwargs['group'] = moe_group
        self.experts = experts
        self.experts_fused = True
        self.num_experts = num_expert

    def expert_fn(self, inp, fwd_cnt):
        # import pdb;pdb.set_trace()
        # print("rank:", torch.distributed.get_rank(), "fwd_cnt:", fwd_cnt)
        return self.experts(inp)

    def _set_ep_group(self, group):
        self.moe_group = group

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)

class FmoeMoE(MoE):
    def __init__(self,
                 hidden_size,
                 expert,
                 num_experts=1,
                 ep_size=1,
                 k=1,
                 capacity_factor=1.,
                 eval_capacity_factor=1.,
                 min_capacity=4,
                 use_residual=False,
                 noisy_gate_policy: typing.Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts=True,
                 use_tutel: bool = False,
                 enable_expert_tensor_parallelism: bool = False):

        super(MoE, self).__init__()

        self.use_residual = use_residual
        self.enable_expert_tensor_parallelism = enable_expert_tensor_parallelism
        assert num_experts % ep_size == 0, f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"
        self.ep_size = ep_size
        self.expert_group_name = f"ep_size_{self.ep_size}"
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

        log_dist(
            f'Creating MoE layer with num_experts: {num_experts} | num_local_experts: {self.num_local_experts} | expert_parallel_size: {self.ep_size}',
            [0])

        assert noisy_gate_policy is None or noisy_gate_policy in ['None', 'Jitter', 'RSample'], \
            'Unsupported noisy_gate_policy: ' + noisy_gate_policy

        ep_group = groups._get_expert_parallel_group(self.expert_group_name)
        # ep_rank=torch.distributed.get_rank(ep_group)
        experts = Experts(expert, self.num_local_experts, self.expert_group_name)
        # fmoe_linear_hacked_expert = experts = MlpExpert(expert,ep_rank=ep_rank,num_expert=1)
        # for name, param in experts.named_parameters():
        #     param.allreduce = False
        #     param.group_name = self.expert_group_name

        self.deepspeed_moe = VitFMoE(experts, d_model=hidden_size,
                                        moe_group = ep_group,
                                     )

        # if self.use_residual:
        #     self.mlp = expert
        #     # coefficient is used for weighted sum of the output of expert and mlp
        #     self.coefficient = torch.nn.Linear(hidden_size, 2)

    def set_deepspeed_parallelism(self):
        self._create_process_groups()

    def _create_process_groups(self):
        # Create process group for a layer if needed
        if self.expert_group_name not in groups._get_expert_parallel_group_dict():
            print(f"No existing process group found, creating a new group named: {self.expert_group_name}")
            if (groups.mpu is None) or (not self.enable_expert_tensor_parallelism):
                # Condition 1 - no groups.mpu means no tensor parallelism
                # Condition 2 - disabling expert tensor parallelism on purpose
                groups._create_expert_and_data_parallel(self.ep_size)
            else:
                # expert tensor parallelism is enabled
                groups._create_expert_data_and_model_parallel(self.ep_size, mpu=groups.mpu)
        # Set the group handle for the MOELayer (deepspeed_moe) object
        self.deepspeed_moe._set_ep_group(groups._get_expert_parallel_group(self.expert_group_name))

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states)
        # if self.use_residual:
        #     # Residual MoE
        #     output_mlp = self.mlp(hidden_states)
        #     if type(output_mlp) is tuple:
        #         output_mlp = output_mlp[0]  # Ignore the bias term for now
        #     coef = self.coefficient(hidden_states)
        #     coef = torch.nn.functional.softmax(coef, dim=-1)
        #     output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, None # self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts
