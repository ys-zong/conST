# [MAE]((https://arxiv.org/abs/2111.06377)) implementation

The implementation of MAE is adopted from [here](https://github.com/pengzhiliang/MAE-pytorch), thanks very much!


## Extract features from spot patches
1. Crop the whole histology image to spot patches according to the spatial coordinates, and stack them in `npy` file. Please note that the size of cropped patches should be 224*224.

2. run MAE to extract features
```bash
# Set the path to save images
OUTPUT_DIR='output/'
# path to image for visualization
PATCHES_PATH='/path/to/cropped/patches.npy'
# path to pretrain model
MODEL_PATH='/path/to/pretrain/checkpoint.pth'


python run_mae_extract_feature.py ${PATCHES_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
```

### Pretrained Weights
The pretrained weights can be found using the link from the original repository [here](https://drive.google.com/drive/folders/182F5SLwJnGVngkzguTelja4PztYLTXfa?usp=sharing)