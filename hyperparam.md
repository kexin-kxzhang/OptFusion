## Hyper-parameter settings

| Method           | Criteo            | Avazu             | KDD12            |
|------------------|-------------------|-------------------|------------------|
| General          |                   |                   |                  |
|                  | bs=4096, dim=40  | bs=4096, dim=40  | bs=4096, dim=16 |
|                  | MLP=[1560, 1560, 1560] | MLP=[960, 960, 960] | MLP=[176, 176, 176] |
| FNN              |                   |                   |                  |
|                  | lr=1e-3, l2=3e-6 | lr=1e-3, l2=3e-6 | lr=3e-5, l2=1e-6 |
| IPNN             |                   |                   |                  |
|                  | lr=3e-3, l2=3e-6 | lr=1e-3, l2=3e-6 | lr=3e-5, l2=1e-6 |
| DeepFM           |                   |                   |                  |
|                  | lr=3e-4, l2=1e-6 | lr=3e-4, l2=3e-6 | lr=3e-5, l2=1e-5 |
| DCN              |                   |                   |                  |
|                  | lr=2e-3, l2=3e-6 | lr=1e-4, l2=3e-6 | lr=3e-5, l2=1e-6 |
| xDeepFM          |                   |                   |                  |
|                  | lr=3e-3, l2=3e-7 | lr=2e-3, l2=2e-7 | lr=3e-5, l2=1e-6 |
|                  | cin_dims=[39, 39, 39] | cin_dims=[24, 24, 24] | cin_dims=[11, 11, 11] |
| DCNv2(s)         |                   |                   |                  |
|                  | lr=1e-3, l2=5e-6 | lr=3e-4, l2=3e-6 | lr=3e-5, l2=1e-7 |
| DCNv2(p)         |                   |                   |                  |
|                  | lr=1e-3, l2=5e-6 | lr=3e-4, l2=3e-6 | lr=3e-5, l2=1e-7 |
| EDCN             |                   |                   |                  |
|                  | lr=3e-4, l2=3e-6 | lr=3e-3, l2=1e-9 | lr=1e-3, l2=0    |
| AutoCTR          |                   |                   |                  |
|                  | lr=2e-3, l2=0    | lr=2e-3, l2=0    | lr=3e-4, l2=0    |
| NASRec           |                   |                   |                  |
|                  | lr=7e-4, 0       | lr=2e-3, l2=0    | lr=1e-3, l2=0    |
| AutoFusion-Hard  |                   |                   |                  |
|                  | lr=3e-4, l2=3e-6 | lr=1.00E-03, l2=0 | lr=2e-3, l2=0    |
| AutoFusion-Soft  |                   |                   |                  |
|                  | lr=3e-4, l2=3e-6 | lr=1.00E-07, l2=0 | lr=3e-8, l2=0    |
