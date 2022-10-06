# Meta-MAPG
Source code for ["A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning"](https://arxiv.org/pdf/2011.00382.pdf) (ICML 2021)

## Dependency
Known dependencies are (please also refer to `requirements.txt`):
```
python 3.6.5
pip3.6
virtualenv
numpy>=1.14.0
torch==1.4.0
mujoco_py==2.1.2.14
gym==0.12.5
tensorboardX==1.2
pyyaml==3.12
gitpython==3.0.8
```

## Setup
To avoid any conflict, please install virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/):
```
pip3.6 install --upgrade virtualenv
```
Please note that all the required dependencies will be automatically installed in the virtual environment by running the training script (`_train.sh`).

## Run
First, please change the config argument in `_train.sh`:
```
ipd.yaml (for IPD)
rps.yaml (for RPS)
half_cheetah_dir.yaml (for 2-Agent HalfCheetah)
```

Only for 2-Agent HalfCheetah experiment, it requires downloading meta-train, validation, and test models:
```
cd pretrain_model
wget --no-check-certificate -r 'https://www.dropbox.com/s/f0506749ns88qq1/HalfCheetahDir-v0.zip?dl=0' -O tmp && unzip tmp && rm tmp
```

After changing the config argument, start training by:
```
./_train.sh
```

Lastly, to see the tensorboard logging during training:
```
tensorboard --logdir=logs
```


## Reference
```
@inproceedings{kim21metamapg,
  title     = {A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning},
  author    = {Kim, Dong Ki and Liu, Miao and Riemer, Matthew D and Sun, Chuangchuang and Abdulhai, Marwa and Habibi, Golnaz and Lopez-Cot, Sebastian and Tesauro, Gerald and How, Jonathan},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages     = {5541--5550},
  year      = {2021},
  editor    = {Meila, Marina and Zhang, Tong},
  volume    = {139},
  series    = {Proceedings of Machine Learning Research},
  month     = {18--24 Jul},
  publisher = {PMLR},
  pdf       = {http://proceedings.mlr.press/v139/kim21g/kim21g.pdf},
  url       = {https://proceedings.mlr.press/v139/kim21g.html},
}
```
