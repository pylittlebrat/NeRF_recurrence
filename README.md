make attempts to use pytorch to recurrence NeRF
**Implement original NeRF (10 days)**

 - [x] network reproduction
 
 - [x] location code
 
 - [x] data loading
 
 - [ ] ray rendering
 
 - [ ] Loss calculation
 
 - [ ] Main

**Running and implementing Mip-NeRF (one week)**

**Running and implementing Semantic-NeRF (one week)**

**Learn to reproduce Block-NeRF (two weeks)**

9-3 NeRF net
# NeRF——Net（pytorch）

代码对网络的实现：
![在这里插入图片描述](https://img-blog.csdnimg.cn/c44ea3ae4fef4a1da6f6f0bb855d6fb0.png)

前八层可视为 网络骨干部分backbone，`self.pts_linears`;
后续其实alpha和rgb的实现是平行的；

forward中：
```python
           self.feature_linear = nn.Linear(W,W)
           self.alpha_linear =nn.Linear(W,1)
           self.rgb_linear = nn.Linear(W // 2, 3)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/285e1c0755d64dc5b88d26346dee4f64.png)

9-4 NeRF Positional encoding
