import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import warnings

class NonTrainableParameter(nn.Parameter):
    """A parameter that can't be trained. Requires grad will always be False"""

    def __new__(cls, data=None, requires_grad=False):
        return super().__new__(cls, data=data, requires_grad=False)

    @property
    def requires_grad(self):
        """It doesn't require grad"""
        return False

    @requires_grad.setter
    def requires_grad(self, value):
        """You can't set it to require grad"""
        pass


def register_params(module, weight_signs, bias_signs=None, bias_negative=None):
    module.register_parameter('weight_signs', NonTrainableParameter(weight_signs))
    if bias_signs is not None:
        module.register_parameter('bias_signs', NonTrainableParameter(bias_signs))
    if bias_negative is not None:
        module.register_parameter('bias_negative', nn.Parameter(bias_negative))


def get_signs(module, handle_biases, active_probability=None, keep_signs=False):
    device = module.weight.device
    
    if keep_signs:
        weight_signs = torch.where(module.weight >= 0, 1, -1).to(device)
        module.weight = nn.Parameter(torch.abs(module.weight))
    else:
        weight_signs = torch.randint(0, 2, size=module.weight.size(), dtype=module.weight.dtype, device=device) * 2 - 1
    
    if active_probability is not None:
        w = torch.abs(module.weight)
        active_tensor = torch.bernoulli(torch.full(w.size(), active_probability, device=device)) * 2 - 1
        module.weight = nn.Parameter(w*active_tensor)

    bias_signs = bias_negative = None
    if module.bias is None or handle_biases == 'ignore':
        return weight_signs, bias_signs, bias_negative

    if handle_biases == 'as_connections':
        if keep_signs:
            weight_signs = torch.where(module.bias >= 0, 1, -1)
        else:
            bias_signs = torch.randint(0, 2, size=module.bias.size(), dtype=module.bias.dtype, device=device) * 2 - 1
        
        if active_probability is not None:
            b = torch.abs(module.weight)
            active_tensor = torch.bernoulli(torch.full(b.size(), active_probability, device=device)) * 2 - 1
            module.bias = nn.Parameter(b*active_tensor)

    elif handle_biases == 'second_bias':
        bias_negative = -module.bias.detach().clone().to(device)
        with torch.no_grad():
            mask = module.bias >= 0
            module.bias[mask] *= 2
            bias_negative[~mask] *= 2

    return weight_signs, bias_signs, bias_negative


def convert_to_deep_rewireable(module: nn.Module, handle_biases="second_bias", active_probability=None, keep_signs=False):
    """Change the forward pass of a standard network to the rewire-forward pass.
       First returns params to be optimized by specific opimizer and then other paramteters"""

    if active_probability is not None and keep_signs:
        warnings.warn('chosen active_probability will be ignored due to keep_signs being active.') 

    sparse_params = []
    other_params = []

    def linear_forward(x, mod=module):
        if handle_biases == 'ignore':
            return F.linear(x, F.relu(mod.weight) * mod.weight_signs, mod.bias)
        elif handle_biases == 'as_connections':
            bias = F.relu(mod.bias) * mod.bias_signs if mod.bias is not None else None
            return F.linear(x, F.relu(mod.weight) * mod.weight_signs, bias)
        elif handle_biases == 'second_bias':
            bias = F.relu(mod.bias) - F.relu(mod.bias_negative) if mod.bias is not None else None
            return F.linear(x, F.relu(mod.weight) * mod.weight_signs, bias)

    def conv2d_forward(x, mod=module):

        if handle_biases == 'ignore':
            if mod.padding_mode != 'zeros':
                return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice, mode=mod.padding_mode),
                                F.relu(mod.weight) * mod.weight_signs, mod.bias, mod.stride, _pair(0),
                                mod.dilation, mod.groups)
            return F.conv2d(x, F.relu(mod.weight) * mod.weight_signs, mod.bias, mod.stride,
                            mod.padding, mod.dilation, mod.groups)
 
        elif handle_biases == 'as_connections':
            bias = F.relu(mod.bias) * mod.bias_signs if mod.bias is not None else None
            if mod.padding_mode != 'zeros':
                return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice, mode=mod.padding_mode),
                                F.relu(mod.weight) * mod.weight_signs, bias, mod.stride, _pair(0),
                                mod.dilation, mod.groups)
            return F.conv2d(x, F.relu(mod.weight) * mod.weight_signs, bias, mod.stride,
                            mod.padding, mod.dilation, mod.groups)
        
        elif handle_biases == 'second_bias':
            bias = F.relu(mod.bias) - F.relu(mod.bias_negative)
            if mod.padding_mode != 'zeros':
                return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice, mode=mod.padding_mode),
                                F.relu(mod.weight) * mod.weight_signs, bias, mod.stride, _pair(0),
                                mod.dilation, mod.groups)
            return F.conv2d(x, F.relu(mod.weight) * mod.weight_signs, bias, mod.stride,
                            mod.padding, mod.dilation, mod.groups)

    if isinstance(module, nn.Linear):
        weight_signs, bias_signs, bias_negative = get_signs(module, handle_biases, active_probability=active_probability, keep_signs=keep_signs)
        register_params(module, weight_signs, bias_signs, bias_negative)
        module.forward = linear_forward
        sparse_params.extend([module.weight])
        if module.bias is not None and handle_biases != 'ignore':
            sparse_params.extend([module.bias])
        elif handle_biases == 'ignore':
            other_params.extend([module.bias])

    elif isinstance(module, nn.Conv2d):
        weight_signs, bias_signs, bias_negative = get_signs(module, handle_biases, active_probability=active_probability, keep_signs=keep_signs)
        register_params(module, weight_signs, bias_signs, bias_negative)
        module.forward = conv2d_forward
        sparse_params.extend([module.weight])
        if module.bias is not None and handle_biases != 'ignore':
            sparse_params.extend([module.bias])
        elif handle_biases == 'ignore':
            other_params.extend([module.bias])

    else:
        other_params.extend(module.parameters(recurse=False))

    for _, submodule in module.named_children():
        sparse_p, other_p = convert_to_deep_rewireable(submodule, handle_biases=handle_biases, active_probability=active_probability, keep_signs=keep_signs)
        sparse_params.extend(sparse_p)
        other_params.extend(other_p)

    return sparse_params, other_params


def merge_back(module: nn.Module):
    """Merge the signs back into the parameters"""
    def merge_signs(p_name, s_name):
        p_hierarchy = p_name.split('.')
        s_hierarchy = s_name.split('.')
        obj = module

        for n in s_hierarchy[:-1]:
            obj = getattr(obj, n)
        sign = getattr(obj, s_hierarchy[-1])
        delattr(obj, s_hierarchy[-1])

        obj = module
        for n in p_hierarchy[:-1]:
            obj = getattr(obj, n)
        value = getattr(obj, p_hierarchy[-1])
        setattr(obj, p_hierarchy[-1], torch.nn.Parameter(value.clamp(min=0) * sign))

    parameters_to_merge = [(n[:-6], n) for n in module.state_dict() if n.endswith('_signs')]
    for p_name, s_name in parameters_to_merge:
        merge_signs(p_name, s_name)

    parameters_to_merge = [(n[:-9], n) for n in module.state_dict() if n.endswith('_negative')]
    for p_name, s_name in parameters_to_merge:
        p_hierarchy = p_name.split('.')
        n_hierarchy = s_name.split('.')
        obj = module

        for n in n_hierarchy[:-1]:
            obj = getattr(obj, n)
        neg = getattr(obj, n_hierarchy[-1])
        delattr(obj, n_hierarchy[-1])

        obj = module
        for n in p_hierarchy[:-1]:
            obj = getattr(obj, n)
        value = getattr(obj, p_hierarchy[-1])
        setattr(obj, p_hierarchy[-1], torch.nn.Parameter(value.clamp(min=0) - neg.clamp(min=0)))


def forward_to_standard(module: nn.Module):
    """Change the forward pass of a rewireable network to the standard forward pass"""
    if isinstance(module, nn.Linear):
        module.forward = lambda x, mod=module: F.linear(x, mod.weight, mod.bias)

    elif isinstance(module, nn.Conv2d):
        module.forward = lambda x, mod=module: (
            F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice, mode=mod.padding_mode),
                     mod.weight, mod.bias, mod.stride, _pair(0), mod.dilation, mod.groups)
            if mod.padding_mode != 'zeros' else
            F.conv2d(x, mod.weight, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
        )

    for _, submodule in module.named_children():
        forward_to_standard(submodule)


def convert_from_deep_rewireable(module: nn.Module):
    """Convert a rewireable network to the structure of a standard network"""
    merge_back(module)
    forward_to_standard(module)


if __name__ == '__main__':
    pass
