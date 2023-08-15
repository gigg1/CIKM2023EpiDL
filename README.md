# Epidemiology-aware Deep Learning for Infectious Disease Dynamics Prediction

[TOC]

## Folder Structures

This is the source code for CIKM 2023 short paper `Epidemiology-aware Deep Learning for Infectious Disease Dynamics Prediction (Mutong Liu, Yang Liu, and Jiming Liu)` and the compared baselines. This package include four folders:

-  **DL4Epi-master-Epi**: This is the code for our proposed method Epi-CNNRNN-Res .
-  **colagnn-master-Run-Epi**: This is the code for our proposed method Epi-Cola-GNN.
-  **DL4Epi-master**: This is the original code of the method CNNRNN-Res. For the details, please refer to the paper [Deep Learning for Epidemiological Predictions](https://raw.githubusercontent.com/CrickWu/crickwu.github.io/master/papers/sigir2018.pdf). For the original code and dataset, please reder to the corresponding [github page](https://github.com/CrickWu/DL4Epi).
-  **colagnn-master-Run**: This is the original code of the method Cola-GNN. For the model details, please refer to the paper [Cola-GNN: Cross-location Attention based Graph Neural Networks for Long-term ILI Prediction](https://yue-ning.github.io/docs/CIKM20-colagnn.pdf). For the original code and dataset, please reder to the corresponding [github page](https://github.com/amy-deng/colagnn).



## How to Run

### Experiments for Epi-CNNRNN-Res

To reproduce the experimental results, please run the following command under the path `./DL4Epi-master-Epi/`:

```bash
bash ./sh/grid_CNNRNN_Res_epi.sh ./data/us_hhs/data.txt ./data/us_hhs/ind_mat.txt hhs 1
```

To check the experimental results, please run:

```bash
python log_parser.py
```



### Experiments for Epi-Cola-GNN

To reproduce the experimental results, please run the following command under the path `./colagnn-master-Run-Epi/src/`:

```
bash ./grid_ColaGNN_epi.sh us_hhs us_hhs-adj hhs cola_gnn_epi
```

To check the experimental results, please run:

```bash
python log_parser.py
```



### Experiments for CNNRNN-Res

To reproduce the experimental results, please run the following command under the path `./DL4Epi-master/`:

```bash
bash ./sh/grid_AR.sh ./data/us_hhs/data.txt hhs 1
bash ./sh/grid_GAR.sh ./data/us_hhs/data.txt hhs 1
bash ./sh/grid_VAR.sh ./data/us_hhs/data.txt hhs 1
bash ./sh/grid_CNNRNN_Res2.sh /data/us_hhs/data.txt ./data/us_hhs/ind_mat.txt hhs 1
```

To check the experimental results, please run:

```
python log_parser.py
```



### Experiments for Cola-GNN

To reproduce the experimental results, please run the following command under the path `./colagnn-master-Run/src/`:

```bash
bash ./grid_ColaGNN.sh us_hhs us_hhs-adj hhs
```

To check the experimental results, please run:

```
python log_parser.py
```



## Citation

> Mutong Liu, Yang Liu, and Jiming Liu. 2023. Epidemiology-aware Deep Learning for Infectious Disease Dynamics Prediction. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM ’23), October 21–25, 2023, Birmingham, United Kingdom. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3583780.3615139

