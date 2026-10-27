# 第 2 周 第 1 课:用梯度下降学会 y = 2x + 1(纯 Python,不用任何库)
data = [(x, 2 * x + 1) for x in [-2, -1, 0, 1, 2, 3]]   # 训练数据

w, b = 0.0, 0.0      # 模型参数,一开始什么都不会
lr = 0.01          # 学习率:每一步走多远

for step in range(201):
    # 1) 前向:算预测和损失(均方误差 MSE)
    loss, dw, db = 0.0, 0.0, 0.0
    for x, y in data:
        pred = w * x + b
        err = pred - y
        loss += err ** 2
        # 2) 求梯度:loss 对 w、b 的偏导(手算的链式法则)
        dw += 2 * err * x
        db += 2 * err
    n = len(data)
    loss, dw, db = loss / n, dw / n, db / n
    # 3) 更新:沿梯度反方向走一小步
    w -= lr * dw
    b -= lr * db
    if step % 20 == 0:
        print(f"step {step:3d}  loss={loss:8.4f}  w={w:.3f}  b={b:.3f}")
