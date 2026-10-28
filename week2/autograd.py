# 第 2 周 第 2 课:迷你自动求导(反向传播),思路来自 Karpathy 的 micrograd
import math

class Value:
    """一个会记住"自己是怎么算出来的"的数字"""
    def __init__(self, data, parents=(), op=""):
        self.data = data          # 数值
        self.grad = 0.0           # 最终 loss 对它的导数,反向传播时填
        self._parents = parents   # 由哪些 Value 算出来
        self._backward = lambda: None  # 怎么把梯度传给 parents
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():          # 加法:梯度原样传下去
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():          # 乘法:梯度乘上"对方"的值
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):         # 只支持常数次方
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other

    def backward(self):
        # 1) 拓扑排序:保证处理一个节点前,用到它的节点都已处理完
        order, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    build(p)
                order.append(v)
        build(self)
        # 2) 从 loss 开始,倒着一路用链式法则往回传
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


if __name__ == "__main__":
    # 演示 1:和第 1 课同一个问题,但梯度不再手推
    data = [(x, 2 * x + 1) for x in [-2, -1, 0, 1, 2, 3]]
    w, b = Value(0.0), Value(0.0)
    for step in range(101):
        loss = sum(((w * x + b) - y) ** 2 for x, y in data) * (1 / len(data))
        w.grad, b.grad = 0.0, 0.0     # 梯度会累加,每步先清零
        loss.backward()               # 一句话算出所有梯度
        w.data -= 0.05 * w.grad
        b.data -= 0.05 * b.grad
        if step % 20 == 0:
            print(f"step {step:3d}  loss={loss.data:.4f}  w={w.data:.3f}  b={b.data:.3f}")

    # 演示 2:随便一个复杂式子,自动求导 vs 数值验证
    a, c = Value(1.5), Value(-0.5)
    f = (a * c + a ** 2).tanh() * 3
    f.backward()
    h = 1e-6
    g = lambda a_, c_: 3 * math.tanh(a_ * c_ + a_ ** 2)
    print("df/da 自动:", round(a.grad, 6), " 数值:", round((g(1.5 + h, -0.5) - g(1.5, -0.5)) / h, 6))
    print("df/dc 自动:", round(c.grad, 6), " 数值:", round((g(1.5, -0.5 + h) - g(1.5, -0.5)) / h, 6))
