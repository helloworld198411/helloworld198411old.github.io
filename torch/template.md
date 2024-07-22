## train_and_eval

### 使用epoches


```python
def train_epoch(model, trainer, loss_fn, dataloader, device):
    """
    train一个epoch
    return: 平均loss
    """
    model.train()
    loss_sum = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        trainer.zero_grad()
        y_hat = model(X)
        l = loss_fn(y_hat, y)
        loss_sum += l.item()
        l.backward()
        trainer.step()
    return loss_sum / len(dataloader)
```


```python
def val_epoch(model, loss_fn, dataloader, device):
    """
    eval一个epoch
    return: 评估指标
    """
    model.eval()
    correct = 0.0
    # loss_sum = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            # acc
            preds = y_hat.argmax(1)
	        accuracy = torch.mean((preds == y).float()).item()
            """
            # loss
            l = loss_fn(y_hat, y)
            loss_sum += l.item()
            """
    return correct / len(dataloader.dataset)
    # return loss_sum / len(dataloader)
```


```python
def train_and_eval(epoches, model, trainer, loss_fn, 
                    train_dataloader, val_dataloader, device):
    """
    训练并评估, 每个epoch记录一次metric
    """
    loss_list, acc_list = [], []
    for epoch in tqdm(range(epoches)):
        loss_list.append(
            train_epoch(model, trainer, loss_fn, train_dataloader, device))
        acc_list.append(
            evaluate_epoch(model, loss_fn, val_dataloader, device))
    print(f'loss:{loss_list[-1]:.4f} acc:{acc_list[-1]:.4f}')
    plt.plot(list(range(epoches)), loss_list, label='loss')
    plt.plot(list(range(epoches)), acc_list, label='acc')
    plt.legend(fontsize='small', loc='best')
    plt.grid(True)
    plt.show()
```

### 使用steps


```python
def forward_batch(batch, model, loss_fn, device):
	"""
    forward一个batch
    return: metric
    """
	X, y = batch
	X, y = X.to(device), y.to(device)
	y_hat = model(X)
	l = loss_fn(y_hat, y)
	
	preds = y_hat.argmax(1)
	accuracy = torch.mean((preds == y).float())
	
	return l, acc
```


```python
def train_and_eval(total_steps, valid_steps, 
					model, optimizer, loss_fn, scheduler, 
					train_dataloader, val_dataloader, device):
    """
	total_steps 约等于 len(dataloader) * epochs
	"""
	train_iter = iter(train_dataloader)
    for step in range(total_steps):
		# 读一个batch的数据
		try:
			batch = next(train_iter)
		except StopIteration:
			train_iter = iter(train_dataloader)
			batch = next(train_iter)

		l, acc = forward_batch(batch, model, loss_fn, device)
		batch_loss = l.item()
		batch_accuracy = acc.item()

		l.backward()
		optimizer.step()
		# scheduler.step() # optional
		optimizer.zero_grad()

        if (step + 1) % valid_steps == 0:
			valid_accuracy = val_epoch(val_dataloader, model, loss_fn, device)
			# 可添加画图和输出可视化metric

```

#### val_epoch使用forward_batch简洁代码


```python
def val_epoch(model, trainer, loss_fn, dataloader, device):
    model.eval()
	running_loss, running_accuracy = 0.0, 0.0
    for batch in dataloader:
		with torch.no_grad():
			l, accuracy = forward_batch(batch, model, loss_fn, device)
			running_loss += l.item()
			running_accuracy += accuracy.item()
    return running_accuracy / len(dataloader)
```

## Learning rate scheduler

### lr scheduler 按照不同类型的scheduler在epoch或batch后进行更新

### 带warmup的learning rate随余弦函数变化的scheduler
- Warmup Phase lr从0线性增加init_lr
- Decay Phase lr随cos逐步降低到0
  - 通过调整num_cycles决定经过cos的多少个周期


```python
import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
	optimizer: Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5, # cos的半个周期 lr降到0后不再上升
	last_epoch: int = -1, # 表示从第0个epoch开始
):
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)
```
