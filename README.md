# Neural Network Resilience to Adversarial Attacks

## Introduction

This project focuses on implementing and testing the resilience of neural network models against various adversarial attacks.

## Installation

You can install the necessary dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage

The project consists of several Python scripts:

- `model.py` - defines the neural network architecture, training and testing procedures.
- `defense.py` - defines the defense methods.
- `adversarial_attack.py` - defines the adversarial attack methods.
- `test_project.py` - framework for testing the model's performance.

To train and test the model, run:

```bash
python model.py [args]
```

Arguments:
- `--model-file` - specify the file for loading or storing the model weights.
- `--force-train` - force training even if a model file exists.
- `--num-epochs` - specify the number of epochs for training.
- `--attack` - choose the type of adversarial attack (FGSM, PGD).
- `--defense` - choose the type of defense (mixup, rse).

To test the model's performance, run:

```bash
python test_project.py [args]
```

Arguments:
- project-dir: path to the project directory to test.
- --batch-size: set batch size for testing.
- --num-samples: number of samples for testing randomized networks.

## Contributors

This project was developed by the following members as part of the Data Science Lab course at Université Paris Dauphine for the academic year 2023/2024:

Artur Dandolini Pescador
Caio Jordan Azevedo
Rafael Benatti

## References

The main papers consulted during the development of this project are:

1. Nicholas Carlini and David Wagner. "Towards Evaluating the Robustness of Neural Networks", 2017.
2. Ting-Jui Chang, Yukun He, and Peng Li. "Efficient Two-Step Adversarial Defense for Deep Neural Networks", 2018.
3. Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. "Explaining and Harnessing Adversarial Examples", 2015.
4. Xuanqing Liu, Minhao Cheng, Huan Zhang, and Cho-Jui Hsieh. "Towards Robust Neural Networks via Random Self-Ensemble", 2018.
5. Tianyu Pang, Kun Xu, and Jun Zhu. "Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks", 2020.
