#!/bin/bash

# Run an experiment: follow the example below. The complete list of arguments can be found in options.py
# python train.py --model gan_es_1d  --dataset XXZ_200_BKT --name proveXXZ --training_reg 0 -0.65  --validation_reg -0.65 -0.8 --wandb_proj <your_project> --wandb_account <your_account>  --val_thr 5e-3 --tr_thr 5e-3 --niter 500 --G_lr 1e-2 --D_lr 1e-3 --Lambda 1 --Epsilon 10
# python train.py --model gan_es_1d  --dataset XXZ_100_BKT --name proveXXZ --training_reg 0 -0.65  --validation_reg -0.65 -0.8 --wandb_proj <your_project> --wandb_account <your_account>  --val_thr 5e-3 --tr_thr 5e-3 --niter 500 --G_lr 1e-2 --D_lr 1e-4 --Lambda 1 --Epsilon 10
# python train.py --model gan_es_1d  --dataset XXZ_160_BKT --name proveXXZ --training_reg 0 -0.65  --validation_reg -0.65 -0.8 --wandb_proj <your_project> --wandb_account <your_account>  --val_thr 3e-3 --tr_thr 4e-3 --niter 500 --G_lr 1e-2 --D_lr 1e-4 --Lambda 1 --Epsilon 10
# python train.py --model gan_es_1d  --dataset XXZ_400_BKT --name proveXXZ --training_reg 0 -0.65  --validation_reg -0.65 -0.8 --wandb_proj <your_project> --wandb_account <your_account>  --val_thr 3e-3 --tr_thr 4e-3 --niter 500 --G_lr 1e-2 --D_lr 1e-4 --Lambda 1 --Epsilon 10

# Evaluate the paper's models and plot the results
python evaluate.py --dataset XXZ_100_BKT --isize 64 --name paper
python evaluate.py --dataset XXZ_160_BKT --isize 64 --name paper
python evaluate.py --dataset XXZ_200_BKT --isize 64 --name paper
python evaluate.py --dataset XXZ_400_BKT --isize 64 --name paper
echo "XXZ model evaluated."

python evaluate.py --dataset BH_32_BKT --isize 64 --name paper
python evaluate.py --dataset BH_64_BKT --isize 64 --name paper
python evaluate.py --dataset BH_100_BKT --isize 64 --name paper
python evaluate.py --dataset BH_192_BKT --isize 64 --name paper
python evaluate.py --dataset BH_256_BKT --isize 64 --name paper
echo "BH model evaluated."

python evaluate.py --dataset BH2S_32_BKT --isize 60 --name paper
python evaluate.py --dataset BH2S_64_BKT --isize 60 --name paper
python evaluate.py --dataset BH2S_96_BKT --isize 60 --name paper
python evaluate.py --dataset BH2S_128_BKT --isize 60 --name paper
echo "BH2S model evaluated."

# Produce the plots of losses for the three models as in Fig.4
python plotter.py
exit 0
