#!/bin/bash

# Run paper experiment
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
