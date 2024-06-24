from ViT import PHMViT, ViT, HariraBaniViT
import torch

print("ViT Model")

model = ViT(
    image_size = 32,
    patch_size = 16,
    num_classes = 10,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
)

print(sum(p.numel() for p in model.parameters()))

print("PHMViT Model")

model = PHMViT(
    image_size = 32,
    patch_size = 16,
    num_classes = 10,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
)

print(sum(p.numel() for p in model.parameters()))

print("HariraBaniViT Model")
model = HariraBaniViT(
    image_size = 32,
    patch_size = 16,
    num_classes = 10,
    dim = 768,
    depth = 12,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
)

print(sum(p.numel() for p in model.parameters()))
i = torch.randn(1, 3, 32, 32).cpu()
model = model.cpu()
print(model(i).shape)

