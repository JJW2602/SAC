import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_mlp():
    pass

def from_numpy(data):
    if isinstance(data, dict):
        return {k: from_numpy(v) for k, v in data.items()}
    else:
        data = torch.from_numpy(data)
        if data.dtype == torch.float64:
            data = data.float()
        return data.to(device)
    
def to_numpy(tensor):
    if isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    else:
        return tensor.to("cpu").detach().numpy()




