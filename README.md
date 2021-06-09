# Meta-MAPG
Source code for ["A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning"](https://arxiv.org/pdf/2011.00382.pdf) (ICML-2021)

## Dependency
Known dependencies are (please also refer to `requirements.txt`):
```
python 3.6.5
pip3.6
virtualenv
numpy>=1.14.0
torch==1.4.0
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
To start training in IPD:
```
./_train.sh
```

To start training in RPS, please change the config argument from `ipd.yaml` to `rps.yaml` in `_train.sh`.

Additionally, to see the tensorboard logging during training:
```
tensorboard --logdir=logs
```

## Todo
- [ ] Upload 2-Agent HalfCheetah code

## Reference
```
@article{kim21metamapg,
  author    = {Dong{-}Ki Kim and Miao Liu and Matthew Riemer and Chuangchuang Sun and Marwa Abdulhai and Golnaz Habibi and Sebastian Lopez{-}Cot and Gerald Tesauro and Jonathan P. How},
  title     = {A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2021},
  url       = {http://arxiv.org/abs/2011.00382}
}
```
