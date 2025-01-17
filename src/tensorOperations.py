from typing import Union, Tuple
import torch
from torch.nn import functional as torchf
from scipy.signal import windows


class MaxOccurrencePool(torch.nn.Module):
    def __init__(
            self,
            kernel_size: int = 3,
            minlength: int = 3,
            beta: float = 1e10,
            device: Union[torch.device, int] = torch.device('cpu')
    ):
        super(MaxOccurrencePool, self).__init__()
        self.kernel_size = kernel_size
        self.minlength = minlength
        self.beta = beta
        self.device = device

    def _softargmax(self, count: torch.Tensor) -> torch.Tensor:
        x_range = torch.arange(count.shape[-1], dtype=count.dtype).to(self.device)
        return torch.sum(torch.nn.functional.softmax(count * self.beta, dim=-1) * x_range, dim=-1)

    @staticmethod
    def postcorrection(x: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        if x.dtype not in [torch.float16, torch.float32, torch.float64]:
            return x
        density_mask = density != 0
        x[density_mask] /= density[density_mask]
        return x

    def forward(
            self,
            x: torch.Tensor,
            return_density: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if len(x.shape) != 3:
            raise ValueError('Dimension of input must be 3')
        x_ = x.transpose(1, 2)
        transpose_dim = x_.shape
        x_ = torchf.pad(x_.type(torch.double), (self.kernel_size // 2 - 1, self.kernel_size // 2), 'constant', 0)
        x_ = x_.unfold(2, self.kernel_size, 1)
        count = torch.stack(
            [torch.zeros(transpose_dim),
             *[torch.sum((x_ == v), dim=-1) for v in torch.arange(1, self.minlength)]],
            dim=-1
        )
        count_mask = torch.zeros_like(count, dtype=torch.bool).to(self.device)
        count_mask[..., 0] = True
        count_mask[torch.any(count > 0, dim=-1)] = False
        count[count_mask] = 1
        # Make argmax differentiable
        x_ = self._softargmax(count)
        x_ = x_.transpose(1, 2).type(x.dtype)
        if return_density:
            return x_, torch.max(count, dim=-1).values.transpose(1, 2)
        else:
            return x_


def get_smoothing_window(smoothing_size: int, smoothing_window: str = 'hann', direction='', **kwargs) -> torch.Tensor:
    if smoothing_window is None or smoothing_window == '':
        window_data = torch.ones(smoothing_size) / float(smoothing_size)
    else:
        window_data = torch.tensor(windows.get_window((smoothing_window, *kwargs.values()), smoothing_size))
        if direction.lower() == 'forward':
            window_data = torch.roll(window_data, shifts=smoothing_size // 2)
            window_data[smoothing_size // 2:] = 0.
        elif direction.lower() == 'backward':
            window_data = torch.roll(window_data, shifts=smoothing_size // 2)
            window_data[:smoothing_size // 2] = 0.
        else:
            pass
        window_data /= torch.sum(window_data)

    return window_data


def create_tensor_kernel(
        smoothing_size: int,
        smoothing_window: str = 'hann',
        do_3d: bool = False,
        direction='',
        **kwargs
) -> torch.nn.Conv1d:
    smoothing_size = smoothing_size if smoothing_size % 2 == 1 else smoothing_size + 1
    if do_3d:
        smoothing_kernel = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(smoothing_size, 3),
            padding=(smoothing_size // 2, 3),
            bias=False
        )
        smoothing_data = torch.zeros((smoothing_size, 3), dtype=torch.double)
        smoothing_data[:, 1] = torch.tensor(get_smoothing_window(
            smoothing_size,
            smoothing_window,
            direction,
            **kwargs
        ))
    else:
        smoothing_kernel = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=(smoothing_size,),
            padding=smoothing_size // 2,
            bias=False
        )
        smoothing_data = get_smoothing_window(smoothing_size, smoothing_window, direction, **kwargs)

    smoothing_kernel.weight.data = smoothing_data.reshape(smoothing_kernel.weight.data.shape)
    smoothing_kernel.requires_grad_(False)
    return smoothing_kernel


def smooth_tensor(
        x: torch.Tensor,
        smoothing_kernel: Union[torch.nn.Conv1d, torch.nn.Conv2d],
        do_append_data: bool = True,
        device: Union[torch.device, int] = torch.device('cpu')
) -> torch.Tensor:
    if len(x.shape) > 3:
        raise ValueError('Only 2 or 3 dimensional tensors are supported')
    if len(x.shape) == 3 and not isinstance(smoothing_kernel, torch.nn.Conv2d):
        raise ValueError('Expect for three dimensional input a Conv2d kernel')
    if len(x.shape) == 2 and not isinstance(smoothing_kernel, torch.nn.Conv1d):
        raise ValueError('Expect for two dimensional input a Conv1d kernel')
    smoothing_size = smoothing_kernel.kernel_size[0]
    # Take into account boundary effects
    x_shape = (*x.shape[:-1], x.shape[-1] + 2 * smoothing_size)
    x_reshape = (x_shape[0], 1, x_shape[1]) if len(x.shape) == 2 else (x_shape[0], 1, x_shape[1], x_shape[2])
    sm_tensor = torch.zeros(x_shape, dtype=torch.double).to(device)
    if do_append_data:
        sm_tensor[..., :smoothing_size] = torch.flip(x[..., :smoothing_size], dims=(1, ))
    sm_tensor[..., smoothing_size:-smoothing_size] = x
    if do_append_data:
        sm_tensor[..., -smoothing_size:] = torch.flip(x[..., -smoothing_size:], dims=(1, ))
    if isinstance(smoothing_kernel, torch.nn.Conv2d):
        sm_tensor = smoothing_kernel(sm_tensor.reshape(x_reshape))[
                    ..., smoothing_size + x.shape[1] // 2:-smoothing_size - x.shape[1] // 2].reshape(x.shape)
    else:
        sm_tensor = smoothing_kernel(sm_tensor.reshape(x_reshape))[
                    ..., smoothing_size:-smoothing_size].reshape(x.shape)
    return sm_tensor


