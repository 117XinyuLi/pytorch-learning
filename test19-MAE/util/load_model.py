import os
import torch
from models_vit import vit_base_patch16
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_


def load_vit_base_patch16(num_classes, save_path, MAE_path, transfer_path, device):

    model = vit_base_patch16(num_classes=num_classes, global_pool=False)

    if os.path.exists(transfer_path):
        checkpoint = torch.load(transfer_path, map_location='cpu')
        print("Load model from", transfer_path)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
        # load pre-trained model
        model.load_state_dict(checkpoint_model, strict=False)
        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    elif os.path.exists(save_path):
        model = model.to(device)
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
                                         model.head)
        model.load_state_dict(torch.load(save_path))
        print('Load model from', save_path)

    elif os.path.exists(MAE_path):
        checkpoint = torch.load(MAE_path, map_location='cpu')
        print("Load model from", MAE_path)
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint)
        # load pre-trained model
        model.load_state_dict(checkpoint, strict=False)
        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)
        # hack: revise model's head with BN
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
                                         model.head)

    model.to(device)
    return model
