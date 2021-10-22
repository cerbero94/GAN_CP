#!/bin/bash

# Run paper experiment
# python train.py --model gan_es_1d  --dataset XXZ_200_BKT_new  --training_reg 0 -0.65  --validation_reg -0.65 -0.8 --wandb_proj gan --wandb_account cerbero94  --val_thr 5e-3 --tr_thr 5e-4 --niter 400 --G_lr 1e-3 --D_lr 1e-5 




# python evaluate.py --dataset BH_32_BKT --isize 64 --validation_reg 2.5 3. --training_reg 0. 2.5 --name paper
# python evaluate.py --dataset BH_64_BKT --isize 64 --validation_reg 2.5 3. --training_reg 0. 2.5 --name paper
# python evaluate.py --dataset BH_100_BKT --isize 64 --validation_reg 2.5 3. --training_reg 0. 2.5 --name paper
# python evaluate.py --dataset BH_192_BKT --isize 64 --validation_reg 2.5 3. --training_reg 0. 2.5 --name paper
# python evaluate.py --dataset BH_256_BKT --isize 64 --validation_reg 2.5 3. --training_reg 0. 2.5 --name paper
# echo "BH model evaluated."

python evaluate.py --dataset XXZ_100_BKT --isize 64 --validation_reg -0.7 -0.8 --training_reg 0. -0.7 --name paper
# python evaluate.py --dataset XXZ_160_BKT --isize 64 --validation_reg -0.7 -0.8 --training_reg 0. -0.7 --name paper
# python evaluate.py --dataset XXZ_200_BKT --isize 64 --validation_reg -0.7 -0.8 --training_reg 0. -0.7 --name paper
# python evaluate.py --dataset XXZ_400_BKT --isize 64 --validation_reg -0.7 -0.8 --training_reg 0. -0.7 --name paper


exit 0
