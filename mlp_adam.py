# 第 2 周 第 3 课:用自己写的 autograd 搭一个小神经网络,对比 SGD 和 Adam
import math, random
from autograd import Value

class Neuron:
    def __init__(self, n_in, act=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.act = act
    def __call__(self, xs):
        s = sum((wi * xi for wi, xi in zip(self.w, xs)), self.b)
        return s.relu() if self.act else s
    def params(self):
        return self.w + [self.b]

class MLP:
    """多层感知机:每层若干神经元,层与层全连接"""
    def __init__(self, sizes):            # 例如 [2, 8, 8, 1]
        self.layers = [[Neuron(sizes[i], act=(i < len(sizes) - 2)) for _ in range(sizes[i + 1])]
                       for i in range(len(sizes) - 1)]
    def __call__(self, xs):
        for layer in self.layers:
            xs = [n(xs) for n in layer]
        return xs[0]
    def params(self):
        return [p for layer in self.layers for n in layer for p in n.params()]

class SGD:
    def __init__(self, params, lr):
        self.params, self.lr = params, lr
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [0.0] * len(params)   # 梯度的移动平均:方向(动量)
        self.v = [0.0] * len(params)   # 梯度平方的移动平均:每个参数自己的步长尺度
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)   # 偏差修正:开头几步 m、v 偏小
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

# 任务:判断点在圆内(+1)还是圆外(-1),直线分不开,必须用非线性的神经网络
random.seed(0)
pts = [(random.uniform(-2, 2), random.uniform(-2, 2)) for _ in range(60)]
data = [((x, y), 1.0 if x * x + y * y < 1.5 else -1.0) for x, y in pts]

def train(opt_name, lr, steps=150):
    random.seed(42)                         # 两种优化器从同一个初始网络出发
    model = MLP([2, 8, 8, 1])
    opt = SGD(model.params(), lr) if opt_name == "SGD" else Adam(model.params(), lr)
    for step in range(steps + 1):
        loss = sum((model(xs) - y) ** 2 for xs, y in data) * (1 / len(data))
        for p in model.params():
            p.grad = 0.0
        loss.backward()
        opt.step()
        if step % 30 == 0:
            acc = sum((model(xs).data > 0) == (y > 0) for xs, y in data) / len(data)
            print(f"{opt_name:4s} step {step:3d}  loss={loss.data:.4f}  准确率={acc:.0%}")

print(f"参数个数: {len(MLP([2, 8, 8, 1]).params())}")
train("SGD", lr=0.05)
print()
train("Adam", lr=0.05)
