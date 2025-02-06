### Simple Transformer Implemented by Pytorch
---
#### 1. Installation
```
pip install -r requirements.txt
```
---
#### 2. Usage
```python
# create model
model = Transformer(vocab_size=vocab_size, d_model=d_model, d_ff=d_ff, n_head=n_head, n_layers=n_layers, dropout=dropout)

# train model in common way
for epoch in range(epochs):
    for batch in train_loader:
        ...

```
---
#### 3. Reference
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 6000-6010).

[2] https://zhuanlan.zhihu.com/p/435782555

