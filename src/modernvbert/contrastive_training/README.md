# Constrastive Training 📄
Here you can find the code to launch a contrastive training for a model using the `colpali_engine` trainer.

> The modeling of `colmodernvbert` and `bimodernvbert` can be found on [the branch `vbert` of `colpali_engine`](https://github.com/illuin-tech/colpali/tree/vbert). We will merge it to the `main` and release a new version of the package soon. In the meantime, please switch to this branch and install from source.

## Usage

### Training
You can launch a training using our script as:
```bash
python train.py --config <config_path>
```

See `./modernvbert` and `./ablation` for config examples.

> You can use your own dataset by creating the loading logic to a `ColPaliEngineDataset` by editing the file `loaders.py` and add a condition to `load_data` of `DatasetArgs` in `config.py`.


### Evaluation
You can evaluate trained models with `mteb` by [using our fork](https://github.com/paultltc/mteb-vlm/tree/vbert). We provide a script to evaluate the models trained from our scripts as:

```bash
python evaluate.py --config <config_path>
```
