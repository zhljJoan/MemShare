对于热点数据的为k的通信比例是

假设$y$作为跨分区访问的比例

$N_{root}^k\cdot \eta_1\cdot m \cdot d_{mem} + (3d_{mem} + d_{e} + d_{n})\cdot \theta \cdot y \cdot N_{nei} $

选择k带来的y可以先暂时估算为
$((|E|-|E^k|)*y+|E^k|*\frac{m-1}{m})/|E|$

化简一下远程访问的比例大概是$(1-|E^k|/|E|)*y + |E^k|/E*\frac{m-1}{m}$


那么通信量可以表示为$|E^k|\cdot \eta_1\cdot m \cdot d_{mem} + d_{fetch}\cdot N_{nei} \cdot \theta \cdot ((1-|E^k|/|E|)+ |E^k|/E*\frac{m-1}{m})$

化简后可以表示为
$|E^k|(\eta_1\cdot m \cdot d_{mem} - d_{fetch} N_{nei}\cdot \theta/|E|) + d_{fetch} \cdot N_{nei} \cdot \theta$

那么$\eta_1 < d_{fetch} N_{nei} \cdot \theta/|E|/({d_{mem} \cdot m}) $带来优势

$\eta_1 < \frac{3}{m}N_{nei}/|E|$则能够减少通信的开销，实验验证当$\alpha$大于0.7的时候，$\eta_1$小于0.1



从精度角度考虑，采用k共享之后，能够采样得到的本地特征的增加数量

对于热点数据,访问外部分区的数量
$|E_k| * \frac{(|E|-|E_k|)}{|E|}\frac{m-1}{m}$

对于非热点数据，访问外部分区的数量
$(|E|-|E_k|)*\frac{|E|-|E_k|}{E}y$
