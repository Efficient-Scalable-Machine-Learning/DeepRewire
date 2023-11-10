import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


class NonTrainableParameter(nn.Parameter):

	def __new__(cls, *args, **kwargs):
		return super().__new__(cls, *args, **kwargs, requires_grad=False)

	@property
	def requires_grad(self):
		return False

	# cant set grad
	@requires_grad.setter
	def requires_grad(self, value):
		pass


def add_signs(module: nn.Module):
	
	# for each parameter of any submodule make a tensor filled with {-1 or 1} and register it as an non-optimizable parameter
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

	if isinstance(module, nn.Linear):
		def linear_forward(x, mod=module):
			return F.linear(x, F.relu(mod.weight)*mod.weight_signs, F.relu(mod.bias)*mod.bias_signs)
		module.forward = linear_forward
	
	elif isinstance(module, nn.Conv2d):
		def conv2d_forward(x, mod=module):
			if mod.padding_mode != 'zeros':
				return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice, mode=mod.padding_mode),
							F.relu(mod.weight)*mod.weight_signs,
							F.relu(mod.bias)*mod.bias_signs, mod.stride, _pair(0), mod.dilation, mod.groups)
			return F.conv2d(x, F.relu(mod.weight)*mod.weight_signs,
								   F.relu(mod.bias)*mod.bias_signs, mod.stride,
								   mod.padding, mod.dilation, mod.groups)
		module.forward = conv2d_forward
	
	for name, submodule in module.named_children():
		forward_to_rewireable(submodule)


def merge_sign_and_weight(module: nn.Module):
	"""
	evil code ahead
	"""
	parameters_to_merge = [(name[:-6], name) for name in module.state_dict() if name.endswith('_signs')]
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

def forward_to_standard(module: nn.Module):

	if isinstance(module, nn.Linear):
		def linear_forward(x, mod=module):
			return F.linear(x, mod.weight, mod.bias)
		module.forward = linear_forward
	
	elif isinstance(module, nn.Conv2d):
		def conv2d_forward(x, mod=module):
			if mod.padding_mode != 'zeros':
				return F.conv2d(F.pad(x, mod._reversed_padding_repeated_twice, mode=mod.padding_mode),
							mod.weight,
							mod.bias, mod.stride, _pair(0), mod.dilation, mod.groups)
			return F.conv2d(x, mod.weight, mod.bias, mod.stride, mod.padding, mod.dilation, mod.groups)
		module.forward = conv2d_forward
	
	for name, submodule in module.named_children():
		forward_to_standard(submodule)



def convert_to_deep_rewireable(module: nn.Module):
	add_signs(module)
	forward_to_rewireable(module)

def convert_from_deep_rewireable(module: nn.Module):
	merge_sign_and_weight(module)
	forward_to_standard(module)

if __name__ == '__main__':

	class TestFCN(nn.Module):
		def __init__(self):
			super(TestFCN, self).__init__()
			self.fc1 = nn.Linear(in_features=500, out_features=200)
			self.fc2 = nn.Linear(in_features=200, out_features=50)
			self.fc3 = nn.Linear(in_features=50, out_features=1)
			self.relu = nn.ReLU()

		def forward(self, x):
			x = self.relu(self.fc1(x))
			x = self.relu(self.fc2(x))
			x = self.fc3(x)
			return x
	
	model = TestFCN()
	convert_to_deep_rewireable(model)
	convert_from_deep_rewireable(model)
	X = torch.rand(100, 500)
	model(X)
	print([(name, p) for name, p in model.named_parameters()])
