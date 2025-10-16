# Modality Alignment 👁️
This folder presents how to align a text encoder with a vision model with [our fork of `m4`](https://github.com/paultltc/smollm/tree/main/vision/m4). 

> Please note that there is no instruction for installation on their repository, so we provided a [requirement.txt](https://github.com/paultltc/smollm/blob/main/vision/m4/requirements.txt) for convinence. Be aware that the versions might not be stable on all environments.

### 1. Alignment pretraining
**From m4 repository**:
You can launch a modality alignment training from a config as follows:
```bash
python m4/training/main.py --config <config_path>
```
See `./modernvbert` and `./ablation` for config examples.

### 2. Gather model and push to the hub
Once your model is trained, you can merge it and push it to the hub as follows
```bash
python merge_and_push.py --checkpoint_dir <path> --repo_id <my-org/my-vbert>
```