import torch
pretrained_dict = torch.load("model_data/swin_s3_base_224.pth")

# change the key of pretrained_dict-> add "backbone.features." for swintransformerv2

updated_pretrained_dict = {}

# Iterate over the keys and modify them as necessary
for key in pretrained_dict.keys():
    new_key = 'backbone.features.' + key
    updated_pretrained_dict[new_key] = pretrained_dict[key]

torch.save(updated_pretrained_dict, 'model_data/swin_s3_base_224_backbone.pth')

new_dict = torch.load("model_data/swin_s3_base_224_backbone.pth")
print(list(new_dict.keys())[:10])