import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 16, lora_alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        self.weight = original_linear.weight
        self.weight.requires_grad = False
        if original_linear.bias is not None:
            self.bias = original_linear.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        
        self.merged = False

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            base_out = F.linear(x, self.weight, self.bias)
            lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return base_out + lora_out
        else:
            return F.linear(x, self.weight, self.bias)

    def merge(self):
        if self.r > 0 and not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += delta_w.to(self.weight.dtype)
            self.merged = True

    def unmerge(self):
        if self.r > 0 and self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= delta_w.to(self.weight.dtype)
            self.merged = False


class LoRAConv1d(nn.Module):
    def __init__(self, original_conv: nn.Conv1d, r: int = 16, lora_alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        self.weight = original_conv.weight
        self.weight.requires_grad = False
        if original_conv.bias is not None:
            self.bias = original_conv.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        if r > 0 and self.groups == 1:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_channels))
            self.lora_B = nn.Parameter(torch.zeros(self.out_channels, r))
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()
        else:
            self.r = 0

        self.merged = False

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            base_out = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            if self.kernel_size == (1,) or self.kernel_size == 1:
                lora_weight = (self.lora_B @ self.lora_A).unsqueeze(-1) * self.scaling
                lora_out = F.conv1d(self.dropout(x), lora_weight)
                return base_out + lora_out
            return base_out
        else:
            return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def merge(self):
        if self.r > 0 and not self.merged and (self.kernel_size == (1,) or self.kernel_size == 1):
            delta_w = (self.lora_B @ self.lora_A).unsqueeze(-1) * self.scaling
            self.weight.data += delta_w.to(self.weight.dtype)
            self.merged = True

    def unmerge(self):
        if self.r > 0 and self.merged and (self.kernel_size == (1,) or self.kernel_size == 1):
            delta_w = (self.lora_B @ self.lora_A).unsqueeze(-1) * self.scaling
            self.weight.data -= delta_w.to(self.weight.dtype)
            self.merged = False


class LoRAConvTranspose1d(nn.Module):
    def __init__(self, original_conv: nn.ConvTranspose1d, r: int = 16, lora_alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.output_padding = original_conv.output_padding
        self.groups = original_conv.groups
        self.dilation = original_conv.dilation
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        self.weight = original_conv.weight
        self.weight.requires_grad = False
        if original_conv.bias is not None:
            self.bias = original_conv.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        if r > 0:
            k = self.kernel_size[0]
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_channels))
            self.lora_B = nn.Parameter(torch.zeros(self.out_channels * k, r))
            self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
            self.reset_parameters()

        self.merged = False

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            base_out = F.conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
            k = self.kernel_size[0]
            delta_w = (self.lora_B @ self.lora_A).view(self.in_channels, self.out_channels, k).permute(0, 1, 2) * self.scaling
            lora_out = F.conv_transpose1d(self.dropout(x), delta_w, None, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
            return base_out + lora_out
        else:
            return F.conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

    def merge(self):
        if self.r > 0 and not self.merged:
            k = self.kernel_size[0]
            delta_w = (self.lora_B @ self.lora_A).view(self.in_channels, self.out_channels, k) * self.scaling
            self.weight.data += delta_w.to(self.weight.dtype)
            self.merged = True

    def unmerge(self):
        if self.r > 0 and self.merged:
            k = self.kernel_size[0]
            delta_w = (self.lora_B @ self.lora_A).view(self.in_channels, self.out_channels, k) * self.scaling
            self.weight.data -= delta_w.to(self.weight.dtype)
            self.merged = False


class LLVCAdapterManager:
    @staticmethod
    def apply_lora(model: nn.Module, r: int = 16, lora_alpha: float = 32.0, dropout: float = 0.0) -> nn.Module:
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, 'label_embedding'):
            for param in model.label_embedding.parameters():
                param.requires_grad = True

        def _inject(parent_module: nn.Module):
            for child_name, child in list(parent_module.named_children()):
                # Do NOT inject LoRA into time-domain waveform synthesis filters (ConvTranspose1d / InConv)
                # to prevent phase destruction and severe output noise!
                if isinstance(child, nn.Linear):
                    wrapped = LoRALinear(child, r=r, lora_alpha=lora_alpha, dropout=dropout)
                    setattr(parent_module, child_name, wrapped)
                elif isinstance(child, nn.Conv1d) and (child.kernel_size == (1,) or child.kernel_size == 1) and child.groups == 1:
                    wrapped = LoRAConv1d(child, r=r, lora_alpha=lora_alpha, dropout=dropout)
                    setattr(parent_module, child_name, wrapped)
                elif isinstance(child, (nn.ConvTranspose1d, nn.LayerNorm)):
                    continue
                else:
                    _inject(child)

        _inject(model)
        return model

    @staticmethod
    def get_adapter_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
        adapter_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad or 'lora_' in name or 'label_embedding' in name:
                adapter_state[name] = param.data.cpu()
        return adapter_state

    @staticmethod
    def save_adapter(model: nn.Module, path: str, metadata: Optional[dict] = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state_dict = LLVCAdapterManager.get_adapter_state_dict(model)
        payload = {
            'adapter': state_dict,
            'metadata': metadata or {},
            'version': '2.0.0'
        }
        torch.save(payload, path)

    @staticmethod
    def inspect_adapter_rank(path: str) -> Tuple[int, float]:
        """
        Detects rank and alpha from checkpoint file.
        """
        payload = torch.load(path, map_location="cpu")
        meta = payload.get('metadata', {})
        r = meta.get('lora_rank')
        alpha = meta.get('lora_alpha')
        if r is not None and alpha is not None:
            return int(r), float(alpha)
        
        # Detect from tensor shapes
        adapter_state = payload.get('adapter', payload)
        for k, v in adapter_state.items():
            if 'lora_A' in k and hasattr(v, 'shape') and len(v.shape) >= 2:
                detected_r = v.shape[0]
                return detected_r, float(detected_r * 2)
        return 16, 32.0

    @staticmethod
    def inject_and_load(model: nn.Module, path: str, map_location: str = "cpu"):
        """
        Automatically inspects rank, injects LoRA into base model, and loads weights.
        """
        r, alpha = LLVCAdapterManager.inspect_adapter_rank(path)
        print(f"[LLVCAdapterManager] Auto-detected LoRA rank={r}, alpha={alpha} from {path}")
        LLVCAdapterManager.apply_lora(model, r=r, lora_alpha=alpha)
        LLVCAdapterManager.load_adapter(model, path, map_location=map_location)
        return model

    @staticmethod
    def load_adapter(model: nn.Module, path: str, map_location: str = "cpu"):
        payload = torch.load(path, map_location=map_location)
        adapter_state = payload.get('adapter', payload)
        
        model_dict = model.state_dict()
        loaded_count = 0
        for k, v in adapter_state.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    model_dict[k].copy_(v)
                    loaded_count += 1
                else:
                    print(f"Warning: Shape mismatch for {k}: model {model_dict[k].shape} vs adapter {v.shape}")
        print(f"[LLVCAdapterManager] Successfully loaded {loaded_count} adapter parameters.")
        return payload.get('metadata', {})

    @staticmethod
    def merge_all(model: nn.Module):
        count = 0
        for module in model.modules():
            if hasattr(module, 'merge') and callable(module.merge):
                module.merge()
                count += 1
        print(f"[LLVCAdapterManager] Merged {count} LoRA modules into base model.")

    @staticmethod
    def unmerge_all(model: nn.Module):
        count = 0
        for module in model.modules():
            if hasattr(module, 'unmerge') and callable(module.unmerge):
                module.unmerge()
                count += 1
        print(f"[LLVCAdapterManager] Unmerged {count} LoRA modules.")

    @staticmethod
    def print_trainable_parameters(model: nn.Module):
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        percent = 100 * trainable_params / all_params if all_params > 0 else 0
        print(f"Total Params: {all_params:,} | Trainable: {trainable_params:,} ({percent:.2f}%)")
