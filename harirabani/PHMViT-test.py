from ViT import PHMViT, ViT
import torch

print("PHMViT Model")

PHMmodel = PHMViT(
    image_size = 32,
    patch_size = 16,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)

print(sum(p.numel() for p in PHMmodel.parameters()))

model = ViT(
    image_size = 32,
    patch_size = 16,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)

print(sum(p.numel() for p in model.parameters()))

