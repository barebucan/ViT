import torch
import torch.cuda as cuda

num_heads = 4
num_blocks = 4
vec_size = 256
d_k = 256
d_model = d_k * num_heads
dropout_p = 0.3

train_split = 0.9
batch_size = 64
LR = 1e-5
checkpoint = True
n_embed = vec_size
pre_training = True
pos_sin = True
xavier = False
flash_att = True

num_classes = 100
H = 224
W = 224
P = 16
N = int(W*H * P**-2)
C = 3
block_size = N + 1

beta1 = 0.9
beta2 = 0.999
weight_decay = 1e-8
Num_gen = 10
Validation_batches = 10

device = torch.device('cuda' if cuda.is_available() else 'cpu')