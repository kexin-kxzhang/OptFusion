# AutoFusion

### Preprocess Dataset
- For Criteo dataset, run 'python preprocess/criteo2TF.py --store_stat --threshold 2'
- For Avazu dataset, run 'python preprocess/avazu2TF.py --store_stat --threshold 2'
- For KDD12 dataset, run 'python preprocess/kdd2TF.py --store_stat --threshold 2'

### Search
run 'bash search.sh'
e.g., 
python search_stage.py \
  --dataset criteo_2 --model optfusion_search \
  --emb_dim 40 --mlp_dims 1560 1560 1560 \
  --fusion_mode 1 \
  --init_constant 5e-1 \
  --lr_emb 3e-4 --lr_nn 3e-4 --arch_lr 3e-1 --l2_emb 3e-6 --l2_nn 3e-6

### Re-train
run 'bash retrain.sh'
e.g.,
python retrain_stage.py \
  --dataset criteo_2 --model autofusion_train \
  --emb_dim 40 --mlp_dims 1560 1560 1560 \
  --lr_emb 3e-4 --lr_nn 3e-4 --l2_emb 3e-6 --l2_nn 3e-6 \
  --fusion_mode 1 \
  --load 'XXX'

Here 'XXX' indicates the logs dictionary generated during the search stage.
