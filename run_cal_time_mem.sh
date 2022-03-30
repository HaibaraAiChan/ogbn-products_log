#!/bin/bash

model=sage
file=ogbn-products
eval=False
# python calculate_time_mem.py > ${model}/res/${file}_${model}_eval_${eval}_mem.log

# python calculate_compute_efficiency.py > ${model}/res/${file}_${model}_eval_${eval}_eff.log

aggre=mean
aggre=lstm
# pMethod=random
pMethod=range
eval=False

resPath=${model}/res/${aggre}/$pMethod/
mkdir $resPath

python calculate_time_mem.py \
--aggre $aggre \
--selection-method $pMethod \
--save-path ${resPath}/${file}_${model}_eval_${eval}_pseudo_ \
> ${resPath}/${file}_${model}_eval_${eval}_mem.log

python calculate_compute_efficiency.py \
--aggre $aggre \
--selection-method $pMethod \
--save-path ${resPath}/${file}_${model}_eval_${eval}_pseudo_ \
> ${resPath}/${file}_${model}_eval_${eval}_eff.log
