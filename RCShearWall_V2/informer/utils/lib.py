import torch

# utils
def get_detach_from(hiddens):
    return [hidden.detach() for hidden in hiddens]

def list_hiddens2torch_tensor(hiddens):
    re = torch.tensor([])
    for hidden in hiddens:
        re = torch.cat([re, hidden], dim=-1)
    return re

def pretty_time(time, degree=1):
    return str(int(time // 60)) + "m" + str(round(time % 60, degree)) if time > 60 else round(time, degree)
