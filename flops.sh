mkdir ./flops

export PYTHONPATH="/home/Rhossolas.Lee/code/Expedit-SAM:$PYTHONPATH"

cluster=256

for grid_stride in 1 2 4 8
do
for model in b l h
do

python grounded_sam_flops.py \
  --device cuda \
  --grid_stride ${grid_stride} \
  --hourglass_num_cluster ${cluster} \
  --sam_checkpoint vit_${model} \
  > ./flops/expedsam_${cluster}_${model}_cuda_${grid_stride}.txt

done
done