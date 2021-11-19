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

LR: 10 steps warm up and then used the Cosine Annealing (2e-5, 5e-4, 2e-5) and the Optimizer is AdamW

test acc: comming soon !!!!!!
