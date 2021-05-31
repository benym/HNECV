## HNECV

This repository provides a reference implementation of *HNECV* as described in the paper:

>HNECV: Heterogeneous Network Embedding via Cloud model and Variational inference.
>Ming Yuan, LiuQun, Guoyin Wang, Yike Guo.
>CAAI International Conference on Artificial Intelligence. 2021.

The paper has been accepted by CICAI, the full paper is coming soon.

### Basic Usage

If you only want to train the model, you need to specify a certain data set, such as dblp, aminer, yelp

```python
python pytorch_HNECV.py --dataset dblp
```

If you want to understand all the processes of the model, you can execute the following command

```python
python pipline.py --dataset dblp
```

> noted: You can adjust the hyperparameters in pytorch_HNECV.py or pipeline.py according to your needs

### Requirements

- Python ≥ 3.6
- PyTorch ≥ 1.7.1
- scipy ≥ 1.5.2
- scikit-learn ≥ 0.21.3
- tqdm ≥ 4.31.1
- numpy
- pandas
- matplotlib

### How to use your own data set

Your input file must be a adjacency matrix, which can be a **mat** file or other compressed format

If you only have the **edgelist** file, you need to follow the preprocessing method in pipline.py, and rewrite the corresponding semantic random walk code.

> If you run pytorch_HNECV.py directly, You need at least the label file of the node, like the initial file in the `dataset/DBLP/reindex_dblp/` folder

### Citing

If HNECV is useful for your research, please cite the following paper:

```
@inproceedings{yuan2021hnecv,
  title={HNECV: Heterogeneous Network Embedding via Cloud model and Variational inference},
  author={Ming Yuan, Qun Liu, Guoyin Wang, Yike Guo},
  booktitle={CAAI International Conference on Artificial Intelligence},
  year={2021},
  address={Hangzhou}
}
```

