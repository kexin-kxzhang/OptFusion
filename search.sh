# Avazu Dataset
python search_stage.py \
  --dataset avazu_2 --model optfusion_search \
  --emb_dim 40 --mlp_dims 960 960 960 \
  --fusion_mode 1 \
  --init_constant 5e-1 \
  --lr_emb 1e-3 --lr_nn 1e-3 --arch_lr 1e-1 --l2_emb 0 --l2_nn 0

#Criteo Dataset
python search_stage.py \
  --dataset criteo_2 --model optfusion_search \
  --emb_dim 40 --mlp_dims 1560 1560 1560 \
  --fusion_mode 1 \
  --init_constant 5e-1 \
  --lr_emb 3e-4 --lr_nn 3e-4 --arch_lr 3e-1 --l2_emb 3e-6 --l2_nn 3e-6

# KDD Dataset
python search_stage.py \
  --dataset kdd_2 --model optfusion_search \
  --emb_dim 16 --mlp_dims 176 176 176 \
  --fusion_mode 1 \
  --init_constant 5e-1 \
  --lr_emb 3e-3 --lr_nn 3e-3 --arch_lr 1e-2 --l2_emb 0 --l2_nn 0
