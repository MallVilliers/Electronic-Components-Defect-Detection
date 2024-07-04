import torch
from repvgg_advance import get_RepVGG_func_by_name
torch_model = get_RepVGG_func_by_name('RepVGG-A0')(deploy=True, num_classes=5)
checkpoint = torch.load('/home/qiqi/output-all/vatten_deploy.pth')
torch_model.load_state_dict(checkpoint, False)
total = sum([param.nelement() for param in torch_model.parameters()])
print("Number of parameter: %.2fM" % (total/1e6))