python retrain_stage.py \
  --dataset avazu_2 --model optfusion_train \
  --emb_dim 40 --mlp_dims 960 960 960 \
  --lr_emb 1e-7 --lr_nn 1e-7 --l2_emb 0 --l2_nn 0 \
  --fusion_mode 1 \
  --load 'avazu_searched_model_arch'

python retrain_stage.py \
  --dataset criteo_2 --model optfusion_train \
  --emb_dim 40 --mlp_dims 1560 1560 1560 \
  --lr_emb 3e-4 --lr_nn 3e-4 --l2_emb 3e-6 --l2_nn 3e-6 \
  --fusion_mode 1 \
  --load 'criteo_searched_model_arch'

python retrain_stage.py \
  --dataset kdd_2 --model optfusion_train \
  --emb_dim 40 --mlp_dims 176 176 176 \
  --lr_emb 3e-8 --lr_nn 3e-8 --l2_emb 0 --l2_nn 0 \
  --fusion_mode 1 \
  --load 'kdd_searched_model_arch'
