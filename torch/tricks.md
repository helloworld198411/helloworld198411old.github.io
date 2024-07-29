## torch tricks

### set seed
```python
import numpy as np
import torch
import random
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(seed)
```

### Gradient Penalty (gp)
#### 通常与生成对抗网络(GANs)相关联，特别是在WGAN-GP
```python
def gp(model, real_data, fake_data, device, lambda_gp):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)

    disc_interpolates = model(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
gp_loss = gradient_penalty(discriminator, real_data, fake_data, device, lambda_gp)
total_loss = real_loss + fake_loss + gp_loss
total_loss.backward()
```


### Automatic Mixed Precision (自动混合精度 AMP)
```python
with autocast():
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    scaler.scale(loss).backward()
scaler.unscale_(optimizer)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```


### Gradient Clip 梯度剪裁
#### 一般在loss.backward() 和 optimizer.step() 之间使用
1. 绝对值裁剪
```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
```
2. 梯度范数裁剪
```python
 # norm_type 决定使用L1还是L2范数
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=2)
```
3. 自定义梯度裁剪
```python
for param in model.parameters():
    if param.grad is not None:
        param.grad.data.clamp_(min=-clip_value, max=clip_value)
```

### 多GPU训练
#### nn.DataParallel
```python
devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
net = nn.DataParallel(net, device_ids=devices)
for epoch in range(num_epochs):
    for X, y in train_iter:
        trainer.zero_grad()
        X, y = X.to(devices[0]), y.to(devices[0])
        l = loss(net(X), y)
        l.backward()
        trainer.step()
```
#### nn.parallel.DistributedDataParallel
```python
from torch.nn.parallel import DistributedDataParallel as DDP
def init_distributed(rank, world_size):
    torch.distributed.init_process_group(
        backend='nccl', 
        init_method='env://', # 适合单机多GPU
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
# usually rank = 0, world_size = torch.cuda.device_count()
init_distributed(rank, world_size)
model = nnModel().cuda(rank)
model = DDP(model, device_ids=[rank])
for epoch in range(num_epochs):
    for data, target in dataloader:
        data, target = data.cuda(rank), target.cuda(rank)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```


### scheduler learning rate plot
```python
lr_history = []
for _ in range(steps):
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    lr_history.append(current_lr)
plt.plot(lr_history, label='Learning Rate', marker='o')
plt.xlabel('Steps')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)
plt.show()
```
