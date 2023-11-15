"""
functions to convert a FCN or CNN to a rewireable neural network
and vice-versa
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class NonTrainableParameter(nn.Parameter):
    """
    a parameter that can't be trained. Requires grad will always be False
    """

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs, requires_grad=False)

    @property
    def requires_grad(self):
        """
        it doesn't require grad
        """
        return False

    # cant set grad
    @requires_grad.setter
    def requires_grad(self, value):
        """
        you cant set it to require grad
        """
        pass


def add_signs(module: nn.Module):
    """
    add the sign parameters to a standard network
    """
    # for each parameter of any submodule make a tensor filled with
    # {-1 or 1} and register it as an non-optimizable parameter
    params_to_register = []
    for name, param in module.named_parameters():
        if '.' not in name and param.requires_grad:
            signs = torch.randint(0, 2, size=param.size(), dtype=param.dtype) * 2 - 1
            params_to_register.append((name + '_signs', NonTrainableParameter(signs)))

    for name, param in params_to_register:
        module.register_parameter(name, param)

    for name, submodule in module.named_children():
        add_signs(submodule)

def forward_to_rewireable(module: nn.Module):
    """
    change the forward pass of a standard network to the rewire-forward pass
    """
    if isinstance(module, nn.Linear):
        def linear_forward(x, mod=module):
            return F.linear(x, F.relu(mod.weight)*mod.weight_signs, F.relu(mod.bias)*mod.bias_signs)
        module.forward = linear_forward

    elif isinstance(module, nn.Conv2d):
        def conv2d_forward(x, mod=module):
            if mod.padding_mode != 'zeros':
                return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice,
                                      mode=mod.padding_mode),
                            F.relu(mod.weight)*mod.weight_signs,
                            F.relu(mod.bias)*mod.bias_signs, mod.stride, _pair(0),
                                   mod.dilation, mod.groups)
            return F.conv2d(x, F.relu(mod.weight)*mod.weight_signs,
                                   F.relu(mod.bias)*mod.bias_signs, mod.stride,
                                   mod.padding, mod.dilation, mod.groups)
        module.forward = conv2d_forward

    for _, submodule in module.named_children():
        forward_to_rewireable(submodule)


def convert_to_rewireable2(module: nn.Module, handle_biases="ignore"):
    """
    change the forward pass of a standard network to the rewire-forward pass
    """

    # Linear
    if isinstance(module, nn.Linear):
        if handle_biases == 'ignore':
            signs = torch.randint(0, 2, size=module.weight.size(), dtype=param.dtype) * 2 - 1
            module.register_parameter('weight_signs', NonTrainableParameter(signs)))
            module.bias = nn.Parameter(torch.ones_like(module.bias))
            def linear_forward(x, mod=module):
                return F.linear(x, F.relu(mod.weight)*mod.weight_signs, 0)

        elif handle_biases == 'as_connections':
            weight_signs = torch.randint(0, 2, size=module.weight.size(), dtype=param.dtype) * 2 - 1
            bias_signs = torch.randint(0, 2, size=module.bias.size(), dtype=param.dtype) * 2 - 1
            module.register_parameter('weight_signs', NonTrainableParameter(weight_signs)))
            module.register_parameter('bias_signs', NonTrainableParameter(bias_signs)))
            def linear_forward(x, mod=module):
                return F.linear(x, F.relu(mod.weight)*mod.weight_signs, F.relu(mod.bias)*mod.bias_signs)

        elif handle_biases == 'second_bias':
            weight_signs = torch.randint(0, 2, size=module.weight.size(), dtype=param.dtype) * 2 - 1
            bias_negative = -module.bias.detach().clone()
            module.register_parameter('weight_signs', NonTrainableParameter(weight_signs)))
            module.register_parameter('bias_negative', NonTrainableParameter(bias_negative)))
            mask =  module.bias >= 0
            module.bias[mask] *= 2
            module.bias_negative[~mask] *= 2
            def linear_forward(x, mod=module):
                return F.linear(x, F.relu(mod.weight)*mod.weight_signs, F.relu(mod.bias)-F.relu(mod.bias_negative))

        module.forward = linear_forward

    # Conv2d (This is way more than nessecary. Maybe fix using partials)
    elif isinstance(module, nn.Conv2d):
        def conv2d_forward(x, mod=module):
            if mod.padding_mode != 'zeros':
                return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice,
                                      mode=mod.padding_mode),
                            F.relu(mod.weight)*mod.weight_signs,
                            F.relu(mod.bias)*mod.bias_signs, mod.stride, _pair(0),
                                   mod.dilation, mod.groups)
            return F.conv2d(x, F.relu(mod.weight)*mod.weight_signs,
                                   F.relu(mod.bias)*mod.bias_signs, mod.stride,
                                   mod.padding, mod.dilation, mod.groups)
        module.forward = conv2d_forward

        if handle_biases == 'ignore':
            signs = torch.randint(0, 2, size=module.weight.size(), dtype=param.dtype) * 2 - 1
            module.register_parameter('weight_signs', NonTrainableParameter(signs)))
            module.bias = nn.Parameter(torch.ones_like(module.bias))
            def conv2d_forward(x, mod=module):
                if mod.padding_mode != 'zeros':
                    return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice,
                                          mode=mod.padding_mode),
                                F.relu(mod.weight)*mod.weight_signs,
                                0, mod.stride, _pair(0),
                                       mod.dilation, mod.groups)
                return F.conv2d(x, F.relu(mod.weight)*mod.weight_signs,
                                       0, mod.stride,
                                       mod.padding, mod.dilation, mod.groups)


        elif handle_biases == 'as_connections':
            weight_signs = torch.randint(0, 2, size=module.weight.size(), dtype=param.dtype) * 2 - 1
            bias_signs = torch.randint(0, 2, size=module.bias.size(), dtype=param.dtype) * 2 - 1
            module.register_parameter('weight_signs', NonTrainableParameter(weight_signs)))
            module.register_parameter('bias_signs', NonTrainableParameter(bias_signs)))
            def conv2d_forward(x, mod=module):
                if mod.padding_mode != 'zeros':
                    return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice,
                                          mode=mod.padding_mode),
                                F.relu(mod.weight)*mod.weight_signs,
                                F.relu(mod.bias)*mod.bias_signs, mod.stride, _pair(0),
                                       mod.dilation, mod.groups)
                return F.conv2d(x, F.relu(mod.weight)*mod.weight_signs,
                                       F.relu(mod.bias)*mod.bias_signs, mod.stride,
                                       mod.padding, mod.dilation, mod.groups)


        elif handle_biases == 'second_bias':
            weight_signs = torch.randint(0, 2, size=module.weight.size(), dtype=param.dtype) * 2 - 1
            bias_negative = -module.bias.detach().clone()
            module.register_parameter('weight_signs', NonTrainableParameter(weight_signs)))
            module.register_parameter('bias_negative', NonTrainableParameter(bias_negative)))
            mask =  module.bias >= 0
            module.bias[mask] *= 2
            module.bias_negative[~mask] *= 2
            def conv2d_forward(x, mod=module):
                if mod.padding_mode != 'zeros':
                    return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice,
                                          mode=mod.padding_mode),
                                F.relu(mod.weight)*mod.weight_signs,
                                F.relu(mod.bias)-F.relu(mod.bias_negative), mod.stride, _pair(0),
                                       mod.dilation, mod.groups)
                return F.conv2d(x, F.relu(mod.weight)*mod.weight_signs,
                                       F.relu(mod.bias)-F.relu(mod.bias_negative), mod.stride,
                                       mod.padding, mod.dilation, mod.groups)


    for _, submodule in module.named_children():
        forward_to_rewireable(submodule)




def merge_sign_and_weight(module: nn.Module):
    """
    evil code ahead
    """
    parameters_to_merge = [(n[:-6], n) for n in module.state_dict() if n.endswith('_signs')]
    for p_name, s_name in parameters_to_merge:
        p_hierarchy = p_name.split('.')
        s_hierarchy = s_name.split('.')
        obj = module
        sign, value = 0, 0
        for n in s_hierarchy[:-1]:
            obj = getattr(obj, n)
        if hasattr(obj, s_hierarchy[-1]):
            sign = getattr(obj, s_hierarchy[-1])
            delattr(obj, s_hierarchy[-1])

        obj = module
        for n in p_hierarchy[:-1]:
            obj = getattr(obj, n)
        if hasattr(obj, p_hierarchy[-1]):
            value = getattr(obj, p_hierarchy[-1])
            setattr(obj, p_hierarchy[-1], torch.nn.Parameter(value.clamp(min=0)*sign))


def merge_back(module: nn.Module):
    """
    evil code ahead
    """
    # merge signs
    parameters_to_merge = [(n[:-6], n) for n in module.state_dict() if n.endswith('_signs')]
    for p_name, s_name in parameters_to_merge:
        p_hierarchy = p_name.split('.')
        s_hierarchy = s_name.split('.')
        obj = module
        sign, value = 0, 0
        for n in s_hierarchy[:-1]:
            obj = getattr(obj, n)
        if hasattr(obj, s_hierarchy[-1]):
            sign = getattr(obj, s_hierarchy[-1])
            delattr(obj, s_hierarchy[-1])

        obj = module
        for n in p_hierarchy[:-1]:
            obj = getattr(obj, n)
        if hasattr(obj, p_hierarchy[-1]):
            value = getattr(obj, p_hierarchy[-1])
            setattr(obj, p_hierarchy[-1], torch.nn.Parameter(value.clamp(min=0)*sign))

    # merge biases
    parameters_to_merge = [(n[:-9], n) for n in module.state_dict() if n.endswith('_negative')]
    for p_name, s_name in parameters_to_merge:
        p_hierarchy = p_name.split('.')
        n_hierarchy = s_name.split('.')
        obj = module
        sign, value = 0, 0
        for n in n_hierarchy[:-1]:
            obj = getattr(obj, n)
        if hasattr(obj, n_hierarchy[-1]):
            neg = getattr(obj, n_hierarchy[-1])
            delattr(obj, n_hierarchy[-1])

        obj = module
        for n in p_hierarchy[:-1]:
            obj = getattr(obj, n)
        if hasattr(obj, p_hierarchy[-1]):
            value = getattr(obj, p_hierarchy[-1])
            setattr(obj, p_hierarchy[-1], torch.nn.Parameter(value.clamp(min=0)-neg.clamp(min=0)))


def forward_to_standard(module: nn.Module):
    """
    change the forward pass of a rewireable network to the standard forward pass
    """
    if isinstance(module, nn.Linear):
        def linear_forward(x, mod=module):
            return F.linear(x, mod.weight, mod.bias)
        module.forward = linear_forward

    elif isinstance(module, nn.Conv2d):
        def conv2d_forward(x, mod=module):
            if mod.padding_mode != 'zeros':
                return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice,
                                      mode=mod.padding_mode), mod.weight,
                            mod.bias, mod.stride, _pair(0), mod.dilation, mod.groups)
            return F.conv2d(x, mod.weight, mod.bias, mod.stride,
                            mod.padding, mod.dilation, mod.groups)
        module.forward = conv2d_forward

    for _, submodule in module.named_children():
        forward_to_standard(submodule)



def convert_to_deep_rewireable(module: nn.Module):
    """
    converts a standard network to the structure of a rewireable network
    """
    add_signs(module)
    forward_to_rewireable(module)



def convert_from_deep_rewireable(module: nn.Module):
    """
    converts a rewireable network to the structure of a standard network
    """
    merge_sign_and_weight(module)
    forward_to_standard(module)


def convert_from_deep_rewireable2(module: nn.Module):
    """
    converts a rewireable network to the structure of a standard network
    """
    merge_back(module)
    forward_to_standard(module)


if __name__ == '__main__':
    pass
