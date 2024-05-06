# Flour Quality Classification

## Table of Contents

- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Hyper-parameters search](#hyper-parameters-search)
- [Training](#training)
- [Evaluation](#evaluation)
- [Demo](#demo)
- [License](#license)

## Installation

1. Install [Hatch](https://hatch.pypa.io/latest/install/) if not installed on your system:

```console
pip install hatch
```

2. Clone the repo:

```
git clone git@github.com:rlebret/sdsc_flour_quality.git
cd sdsc_flour_quality
```

## Data Preprocessing

Start by splitting the dataset into train/test sets:

```
hatch run python scripts/create_train_test_file.py \
    data/flour_dataset.csv \
    --test-size 0.2 \
    --random-state 42 \
    --output-path data
```

Remove missing and negative values:

```console
mkdir data/preprocessed
```

```
z_threshold=2
hatch run preprocess \
    "data/train.csv" \
    --z-threshold $z_threshold \
    --remove-empty-rows \
    --remove-negative-rows \
    --output-path "data/preprocessed/flour_z_${z_threshold}.csv"
```

Impute missing and negative values:

```console
z_threshold=3
hatch run preprocess \
    "data/train.csv" \
    --z-threshold $z_threshold \
    --impute-missing-values \
    --impute-negative-values \
    --output-path "data/preprocessed/flour_z_${z_threshold}_impute.csv"
```

## Hyper-parameters search

```console
mkdir data/hyperparameters
```

Randomized hyper-parameters search:

```console
scaling="standard"
filename="flour_z_2_impute"
model_name="SVRFlour"
input_filename="data/preprocessed/${filename}.csv"
output_filename="data/hyperparameters/${filename}_${model_name}_${scaling}.json"
hatch run hyperparameters \
    "$input_filename" \
    --scaling_method $scaling \
    --model_name $model_name \
    --output_path "$output_filename" \
    --regression
```

## Training

```console
mkdir checkpoints
```

Train the classification model:

```console
scaling="none"
filename="flour_z_2_impute"
model_name="rf"
hatch run train "configs/${model_name}_${filename}_${scaling}.yaml"
```

## Evaluation

```console
scaling="none"
filename="flour_z_2_impute"
model_name="rf"
hatch run evaluate data/test.csv "configs/${model_name}_${filename}_${scaling}.yaml"
```

## Demo

Run the demo with a chosen model:

```console
scaling="standard"
filename="flour_z_3_impute"
model_name="lr"
hatch run demo:run -- --config-file=configs/${model_name}_${filename}_${scaling}_cv.yaml
```

## License

`flour` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
