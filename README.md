# TOWARDS EXPLAINABLE AI4EO: AN EXPLAINABLE DEEP LEARNING APPROACH FOR CROP TYPE MAPPING USING SATELLITE IMAGES TIME SERIES

## Usage
Setup conda environment and activate (some modules may not be listed in the environment dependencies, install them manually)
```
conda env create -f environment.yml
conda activate timematch
```

Download dataset and extract to `../data/timematch_data` (or set `--data_root` to its path for `train.py`).

Pre-trained models are extracted to `./outputs`.

Positional Encoding representations can be generated using the `./dtw/PE_illustrations.ipynb` notebook.

## Credits
- This repository is based upon the original [Thermal Positional Encoding code ](https://github.com/jnyborg/tpe) and [TimeMatch code](https://github.com/jnyborg/timematch)
- The implementation of PSE+LTAE is based on [the official implementation](https://github.com/VSainteuf/lightweight-temporal-attention-pytorch)
