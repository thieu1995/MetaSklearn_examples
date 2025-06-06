# MetaSklearn Experiments

Seaborn color:

https://www.practicalpythonfordatascience.com/ap_seaborn_palette


## Setup environment

#### Create new environment
```shell
python -m venv pve
```

*) On Windows:
```shell
.\pve\Scripts\activate
```

*) On Linux-based
```shell
source pve/bin/activate
```

#### Install requirement file
```bash
pip install -r requirements.txt
```

## Run scripts to get results

```bash
python 01_cdcd.py
```


# Large-scale dataset

### 1. CDC Diabetes Health Indicators
+ samples: 253680
+ features: 21
+ feature type: Categorical, Integer
+ task: Binary Classification [0, 1]
+ subject: Health and Medicine
+ dataset characteristics: Tabular, Multivariate
+ link: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

### 2. RT-IoT2022
+ samples: 123117
+ features: 83
+ feature type: Real, Categorical
+ task: Multi Classification [0->11], Regression, Clustering
+ subject: Engineering
+ dataset characteristics: Tabular, Sequential, Multivariate
+ link: https://archive.ics.uci.edu/dataset/942/rt-iot2022
+ https://www.kaggle.com/code/azimuddink/azimuddink-rt-iot

### 3. Superconductivty Data
+ samples: 21263
+ features: 81
+ feature type: Integer, Real
+ task: Regression
+ subject: Physics and Chemistry
+ dataset characteristics: Multivariate
+ link: https://archive.ics.uci.edu/dataset/464/superconductivty+data
