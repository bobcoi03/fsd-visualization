import torch

checkpoint = torch.load('./models/culane-resnet-18.pth')
print(checkpoint["model"])

i = 0
for name, param in checkpoint['model'].items():
    print(f"{name}: {param.shape}")