import torch

from metamorph.config import cfg


def restore_from_checkpoint(ac):
    print("Device is:")
    print(cfg.DEVICE)
    MAP_LOCATION = torch.device("cpu") if cfg.DEVICE == "cpu" else None
    print(f"Loading checkpoint from {cfg.PPO.CHECKPOINT_PATH} on {MAP_LOCATION}")
    model_p, ob_rms = torch.load(cfg.PPO.CHECKPOINT_PATH, weights_only=False, map_location=MAP_LOCATION)

    state_dict_c = ac.state_dict()
    state_dict_p = model_p.state_dict()

    fine_tune_layers = set()
    layer_substrings = cfg.MODEL.FINETUNE.LAYER_SUBSTRING
    for name, param in state_dict_c.items():

        if any(name_substr in name for name_substr in layer_substrings):
            fine_tune_layers.add(name)

        if name in state_dict_p:
            param_p = state_dict_p[name]
        else:
            print (f'the model does not have {name}')
            continue

        if param_p.shape == param.shape:
            with torch.no_grad():
                param.copy_(param_p)
        else:
            raise ValueError(
                "Checkpoint path is invalid as there is shape mismatch"
            )

    if not cfg.MODEL.FINETUNE.FULL_MODEL:
        for name, param in ac.named_parameters():
            if name not in fine_tune_layers:
                param.requires_grad = False
            else:
                print (f'fine tune {name}')
    else:
        print ('fine tune the whole model')

    return ob_rms
