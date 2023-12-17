"""Attention module."""

import numpy as np
import torch
import torch.nn.functional as F

import cliport.models as models
import cliport.models.core.fusion as fusion
from cliport.models.core.attention import Attention


class AttentionImageGoal(Attention):
    """Attention (a.k.a Pick) with image-goals module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def forward(self, inp_img, goal_img, softmax=True):
        """Forward pass."""
        # Input image.
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)

        goal_tensor = np.pad(goal_img, self.padding, mode='constant')
        goal_shape = (1,) + goal_tensor.shape
        goal_tensor = goal_tensor.reshape(goal_shape)
        goal_tensor = torch.from_numpy(goal_tensor.copy()).to(dtype=torch.float, device=self.device)
        # in_tens = in_tens * goal_tensor  # originally goal_tensor might be a mask

        # Now goal_tensor is images + depth + fused features, so we just take average of two tensors / or subtract them
        # in_tens = (in_tens + goal_tensor) / 2
        in_tens = in_tens - goal_tensor

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_tens:
            logits.append(self.attend(x))
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # D H W C
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output
    

class AttentionImageGoalFusionLat(AttentionImageGoal):
    """Attention (a.k.a Pick) with image-goals module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)
        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x_one, x_two):
        x1, lat = self.attn_stream_one(x_one)
        x2, _ = self.attn_stream_two(x_two, lat)
        x = self.fusion(x1, x2)
        return x
    
    def forward(self, inp_img, goal_img, softmax=True):
        """Forward pass."""
        # Input image.
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)

        goal_tensor = np.pad(goal_img, self.padding, mode='constant')
        goal_shape = (1,) + goal_tensor.shape
        goal_tensor = goal_tensor.reshape(goal_shape)
        goal_tensor = torch.from_numpy(goal_tensor.copy()).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Rotate goal.
        goal_tensor = goal_tensor.permute(0, 3, 1, 2)
        goal_tensor = goal_tensor.repeat(self.n_rotations, 1, 1, 1)
        goal_tensor = self.rotator(goal_tensor, pivot=pv)

        # Forward pass.
        logits = []
        for x1, x2 in zip(in_tens, goal_tensor):
            lgts = self.attend(x1, x2)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output

class AttentionImageGoalFusion(AttentionImageGoal):
    """Attention (a.k.a Pick) with image-goals module."""

    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)
        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x_one, x_two):
        x1 = self.attn_stream_one(x_one)
        x2 = self.attn_stream_two(x_two)
        x = self.fusion(x1, x2)
        return x
    
    def forward(self, inp_img, goal_img, softmax=True):
        """Forward pass."""
        # Input image.
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)

        goal_tensor = np.pad(goal_img, self.padding, mode='constant')
        goal_shape = (1,) + goal_tensor.shape
        goal_tensor = goal_tensor.reshape(goal_shape)
        goal_tensor = torch.from_numpy(goal_tensor.copy()).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Rotate goal.
        goal_tensor = goal_tensor.permute(0, 3, 1, 2)
        goal_tensor = goal_tensor.repeat(self.n_rotations, 1, 1, 1)
        goal_tensor = self.rotator(goal_tensor, pivot=pv)

        # Forward pass.
        logits = []
        for x1, x2 in zip(in_tens, goal_tensor):
            lgts = self.attend(x1, x2)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])
        return output