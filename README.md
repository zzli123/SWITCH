# SWITCH: deep generative model for spatial multi-omics integration and cross-modal prediction

## Installation
The installation will take approximately 10 minutes.
```
git clone https://github.com/zzli123/SWITCH.git
cd SWITCH-main
conda create --name switch_env python=3.8
conda activate switch_env
pip install -r requirement.txt
```

## Data
Example datasets for tutorials can be downloaded from the following link: [Benchmark datasets for SWITCH](https://zenodo.org/records/15602076)

To facilitate processing of GEO-downloaded raw data, we also provide dedicated scripts in the `preprocess` folder.

## Tutorials
The step-by-step tutorials are included in the `Benchmark/SWITCH` folder to show how to use SWITCH.  Running Tutorial 1 is expected to take around 10 minutes.

- Tutorial 1: Integrating E13 Mouse embryo (Spatial-ATAC-RNA-seq)
- Tutorial 2: Integrating P22 Mouse brain (Spatial CUT&TAG-RNA-seq, RNA+H3K27ac)
- Tutorial 3: Integrating P22 Mouse brain (Spatial CUT&TAG-RNA-seq, RNA+H3K4me3)
- Tutorial 4: Integrating P21 Mouse brain (Spatial-ATAC-RNA-seq)

## Benchmarking
In our study, we compared SWITCH with 9 state-of-the-art single-cell multi-omics integration methods, including Seurat (V3), LIGER, BindSC, GLUE, SCALEX, MaxFuse, SIMBA, scConfluence and Monae, as well as with 4 state-of-the-art cross-modal translation methods, including JAMIE, MultiVI, scButterfly, and Monae. Jupyter notebooks covering the benchmarking analysis in this paper are included in the `Benchmark/others` folder.

## Testing
The `tests` directory contains pytest scripts for automated testing of core functionalities. To run all tests, execute:
```
pytest tests/
```

## Support
If you have any questions, please contact us [zzli@tongji.edu.cn](mailto:zzli@tongji.edu.cn).
