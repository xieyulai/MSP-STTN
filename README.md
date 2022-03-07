# MSP-STTN

Code and data for the paper [Multi-Size Patched Spatial-Temporal Transformer Network for Short- and Long-Term Grid-based Crowd Flow Prediction]()

Please cite the following paper if you use this repository in your research.
```
Under construction
```

Note that this project consists of **four** parts.
1. Data preparation from the original dataset to the required form. ([MSP-STTN-DATA]())
2. Code for BikeNYC dataset. (This repo) 
3. Code for TaxiBJ dataset. ([MSP-STTN-BJ]())
3. Code for CrowdDensityBJ dataset. ([MSP-STTN-DENSITY]())

## TaxiBJ

### Package
```
PyTorch > 1.07
```
Please refer to `requirements.txt`

### Data Preparation
- Processing data according to [MSP-STTN-DATA]().
- The `data\` should be like this:
```bash
data
___ BikeNYC
```
- Or the processed data can be downloaded from [BAIDU_PAN](https://pan.baidu.com/s/11V0XBDXOi4rkxP6YI8VI6A),PW:`a7m7`.

### Pre-trained Models
- Several pre-trained models can be downloaded from [BAIDU_PAN](https://pan.baidu.com/s/1_YO_NVuvCv45p0vG5enZLw), PW:`0f6e`.
- The `model\` should be like this:
```bash
model
___ Imp_0043
___ ___ pre_model_12.pth
___ Imp_0047
___ ___ pre_model_34.pth
___ Imp_0051
___ ___ pre_model_2.pth
___ Imp_104
___ ___ pre_model_46.pth
___ Imp_1051
___ ___ pre_model_3.pth
___ Imp_115
___ ___ pre_model_26.pth
___ Imp_2051
___ ___ pre_model_2.pth
___ Imp_3089
___ ___ pre_model_79.pth
___ Imp_5042
___ ___ pre_model_36.pth
___ Imp_5047
___ ___ pre_model_10.pth
___ Imp_5050
    ___ pre_model_2.pth
```
- Use `sh BEST.sh` for short-term prediction.
- Use `sh BEST_long.sh` for short-term prediction.

### Train and Test
- Use `sh TRAIN.sh` for short-term prediction.
- Use `sh TRAIN_long.sh` for short-term prediction.

### Repo Structure
```bash
___ BEST_long.sh
___ BEST.sh
___ data # Data
___ dataset
___ model # Store the training weights
___ net # Network struture
___ pre_main_long.py # Main function for long-term prediction
___ pre_main_short.py # Main function for shot-term prediction
___ pre_setting_nyc_long.yaml # Configuration for long-term prediction
___ pre_setting_nyc.yaml # Configuration for short-term prediction
___ README.md
___ record # Recording the training and the test
___ test # Record the result under time slots in testing set
___ TRAIN_long.sh
___ TRAIN.sh
___ util
```
