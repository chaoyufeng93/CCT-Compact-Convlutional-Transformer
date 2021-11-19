This model is learning from scratch, I applied the AutoAugment!

CCT-4/3*2:

layer: 4

head: 2

conv_layer: 2

kernel_size: [3, 3]

dim_expan: 1

emb_dim: [64, 128]

dropout: 0.1

batch_size: 128

epoch: 200

LR: 10 steps warm up and then used the Cosine Annealing (2e-5, 5e-4, 2e-5) and the Optimizer is AdamW. And I also applied label smoothing: p = 0.1

test acc: 87.08%

The best epoch of val acc (87.1%) is 194th , I believe if I train much longer, the performance will increase! (the training acc at the final epoch is 90.03%)
