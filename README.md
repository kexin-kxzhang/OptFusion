# Fusion Matters: Learning Fusion in Deep Click-through Rate Prediction Models (WSDM 2025)

This repository contains the official implementation of our WSDM 2025 paper:  
📘: [Fusion Matters: Learning Fusion in Deep Click-through Rate Prediction Models](https://arxiv.org/abs/2411.15731).


If you find this repository helpful to your work, please kindly star it and cite our paper as follows:
```bibtex
@article{zhang2024optfusion,
      title={Fusion Matters: Learning Fusion in Deep Click-through Rate Prediction Models}, 
      author={Kexin Zhang and Fuyuan Lyu and Xing Tang and Dugang Liu and Chen Ma and Kaize Ding and Xiuqiang He and Xue Liu},
      journal={arXiv preprint arXiv:2411.15731},
      year={2024}
}
```
---

### Preprocess Dataset
- For Criteo dataset, run <pre style="background: #f0f0f0; display: inline-block;">python dataprocess/criteo2TF.py --store_stat --threshold 2</pre>
- For Avazu dataset, run <pre style="background: #f0f0f0; display: inline-block;">python dataprocess/avazu2TF.py --store_stat --threshold 2</pre>
- For KDD12 dataset, run <pre style="background: #f0f0f0; display: inline-block;">python dataprocess/kdd2TF.py --store_stat --threshold 2</pre>

### Search
Run <pre style="background: #f0f0f0; display: inline-block;">bash search.sh</pre>
Here, take Criteo dataset as an example:

<pre style="background: #f0f0f0; padding: 10px;">
python search_stage.py \
  --dataset criteo_2 --model optfusion_search \
  --emb_dim 40 --mlp_dims 1560 1560 1560 \
  --fusion_mode 1 \
  --init_constant 5e-1 \
  --lr_emb 3e-4 --lr_nn 3e-4 --arch_lr 3e-1 --l2_emb 3e-6 --l2_nn 3e-6
</pre>

### Re-train
Run <pre style="background: #f0f0f0; display: inline-block;">bash retrain.sh</pre>
Here, take Criteo dataset as an example:

<pre style="background: #f0f0f0; padding: 10px;">
python retrain_stage.py \
  --dataset criteo_2 --model optfusion_train \
  --emb_dim 40 --mlp_dims 1560 1560 1560 \
  --lr_emb 3e-4 --lr_nn 3e-4 --l2_emb 3e-6 --l2_nn 3e-6 \
  --fusion_mode 1 \
  --load 'XXX'
</pre>
Here 'XXX' indicates the logs dictionary generated during the search stage.
