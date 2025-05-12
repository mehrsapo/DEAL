import torch

def periodic_pad_transpose(tensor: torch.Tensor, pad_size) -> torch.Tensor:
        """
        Adjoint of the periodic padding operation which amounts to a special type of cropping.

        :param tensor: input of shape [..., H, W] to be cropped
        :return: cropped tensor of shape [..., H - pad_h, W - pad_w]
        """
        sz = [tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)]
        out = tensor.clone()
        # Top
        if pad_size[1] != 0:
            out[..., pad_size[0]:pad_size[0] + pad_size[1], :] += out[..., sz[-2]-pad_size[1]::, :]
        # Bottom
        if pad_size[0] != 0:
            out[..., -pad_size[0] - pad_size[1]:sz[-2]-pad_size[1], :] += out[..., 0:pad_size[0], :]
        # Left
        if pad_size[3] != 0:
            out[..., pad_size[2]:pad_size[2] + pad_size[3]] += out[..., sz[-1]-pad_size[3]::]
        # Right
        if pad_size[2] != 0:
            out[..., -pad_size[2] - pad_size[3]:sz[-1]-pad_size[3]] += out[..., 0:pad_size[2]]
        if pad_size[1] == 0:
            end_h = sz[-2] + 1
        else:
            end_h = sz[-2] - pad_size[1]
        if pad_size[3] == 0:
            end_w = sz[-1] + 1
        else:
            end_w = sz[-1] - pad_size[3]
        out = out[..., pad_size[0]:end_h, pad_size[2]:end_w]
        return out