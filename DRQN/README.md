# DQN (Deep Recurrent Q Network)

## 相关文献
+ [Deep Recurrent Q-Learning for Partially Observable MDPs](./asset/DRQN.pdf)

## Deep Recurrent Q-Learning for Partially Observable MDPs
+ idea
  + 使用LSTM取代原DQN网络卷积层之后的第一个全连接层，以处理连续输入的帧之间的时序信息