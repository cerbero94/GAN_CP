# GAN_CP
This repository contains PyTorch implementation of the following paper: 
Detection of Berezinskii-Kosterlitz-Thouless transition via Generative Adversarial Networks [[1]](#reference)

The code can be used in general for detecting critical points (CP) of physical systems in an unsupervised fashion. 

The structure of the code is based on [GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training](https://github.com/samet-akcay/ganomaly)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/cerbero94/GAN_CP.git
   ```
2. Create and activate the virtual environment using conda:
    ```
    conda create -n gan_cp python=3.7  
    conda activate gan_cp
    ```
3. Install the dependencies contained in the requirements file:
   ```
   pip install --user --requirement requirements.txt
   ```

## Evaluation of the models in the paper
In order to run the evaluation of the models trained for the paper, execute:
```
./paper_figures.sh
```
It will reproduce the plots of Fig. 4 by loading the pre-trained models. 

## Reference
[1]  D. Contessi and E. Ricci and A. Recati and M. Rizzi (2021) "Detection of Berezinskii-Kosterlitz-Thouless transition via Generative Adversarial Networks", [arXiv:2110.05383][paper]

[paper]: https://arxiv.org/abs/2110.05383