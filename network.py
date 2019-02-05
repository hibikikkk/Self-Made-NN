import numpy as np


class ActivationMethods:

    @staticmethod
    def step(x):
        if 0 >= x:
            return 0
        return 1

    class Relu:
        def __init__(self):
            self.mask = None

        def forward(self, x):
            self.mask = (x <= 0)
            out = x.copy()
            print(out)
            out[self.mask] = 0
            return out

        def backward(self, dout):
            dout[self.mask] = 0
            dx = dout
            return dx

    class Sigmoid:

        def __init__(self):
            self.out = None

        def forward(self, x):
            out = 1 / (1 + np.exp(-x))
            self.out = out

            return out

        def backward(self, dout):
            dx = dout * (1.0 - self.out) * self.out
            return dx

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(np.exp(x))


class LossFunctions:

    @staticmethod
    def cross_entropy_error(y, label):
        dlt = 1e-8
        return -np.sum(label * np.log(y + dlt))

    @staticmethod
    def mean_squared_error(y, label):
        return np.sum((label - y) ** 2) / 2


class NeuralNetwork:

    def __init__(self, input_size, output_size, bias=None, weight=None, act_func=None, output_func=None,
                 loss_func=None):
        """
        input_size:int 入力の数
        node:int ニューロン数
        weight:numpy.array 設定する層の入力サイズとノード数によって多次元配列を入力する。入力がない場合ランダムで決定
        act_func:method or str 活性化関数の設定。設定がない場合活性化関数を使用しない
        """
        self.input_size = input_size
        self.output_size = output_size
        if weight:
            self.weight = weight
        else:
            self.weight = np.random.normal(loc=0, scale=0.1, size=(input_size, output_size))
        if bias:
            self.bias = bias
        else:
            self.bias = np.random.normal(loc=0, scale=0.5, size=1)

        self.act_func = act_func
        self.output_func = output_func
        self.loss_func = loss_func

    def forward(self, x):
        """
        前向き推論
        :param x:numpy.array 入力
        :return: 計算結果
        """
        if self.act_func:
            if type(self.act_func) == "str":
                pass
            else:
                return np.array(list(map(self.act_func.forward, (np.dot(x, self.weight) + self.bias))))

        elif self.output_func:
            return self.output_func((np.dot(x, self.weight) + self.bias))

        else:
            return np.dot(x, self.weight) + self.bias

    def backward(self, dx):
        pass

    def gradient_descent(self, f, init_x, lr=0.01, step_num=100):
        x = init_x

        for i in range(step_num):
            grad = self.numerical_differentiation(f, x)
            x -= lr * grad

        return x

    def numerical_differentiation(self, f, X):
        if X.ndim == 1:
            return self._numerical_no_batch(f, X)
        else:
            grad = np.zeros_like(X)

            for idx, x in enumerate(X):
                grad[idx] = self._numerical_no_batch(f, x)

            return grad

    @staticmethod
    def _numerical_no_batch(f, x):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(x)

        for idx in range(x.size):
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val  # 値を元に戻す
        return grad


class Layer:

    def __init__(self):
        """ネットワーク管理するクラス

        """
        self.input = None
        self.hidden_layer = []
        self.output = None

    def add(self, **kwargs):
        """隠れ層を追加するメソッド

        :param node:int 隠れ層のノード数設定
        :param out:int 出力数の設定
        :param activation:ActivationMethod 活性化関数の設定
        :param output_func:ActivationMethod 出力関数関数の設定
        :return:None
        """
        hidden_out = None
        hidden_act = None
        output_func = None
        for key in kwargs:
            if key == "node":
                hidden_node = kwargs[key]

            if key == "output_num":
                hidden_out = kwargs[key]

            if key == "activation":
                hidden_act = kwargs[key]

            if key == "output_func":
                output_func = kwargs[key]

        if self.hidden_layer:
            self.hidden_layer[len(self.hidden_layer) - 1]["out"] = hidden_node
            self.hidden_layer.append(
                {"node": hidden_node, "out": hidden_out, "activation": hidden_act, "output_func": output_func,
                 "net": None})
        else:
            self.hidden_layer.append(
                {"node": hidden_node, "out": hidden_out, "activation": hidden_act, "output_func": output_func,
                 "net": None})

    def predict(self, x):
        for i, val in enumerate(self.hidden_layer):
            if not self.hidden_layer[i]["net"]:
                self.hidden_layer[i]["net"] = NeuralNetwork(input_size=val["node"], output_size=val["out"],
                                                            act_func=val["activation"], output_func=val["output_func"])
            y = self.hidden_layer[i]["net"].forward(x)

        return y

    def fit(self, x):
        y = self.predict(x)
        for i, val in enumerate(self.hidden_layer):
            self.hidden_layer[i]["net"]


if __name__ == "__main__":
    x = np.array([1, 1, 1, 0, 1, 0])
    layer = Layer()
    layer.add(node=6, activation=ActivationMethods.Sigmoid())
    layer.add(node=6, activation=ActivationMethods.Relu())
    layer.add(node=6, activation=ActivationMethods.Sigmoid())
    layer.add(node=6, activation=ActivationMethods.Relu())
    layer.add(node=6, activation=ActivationMethods.Sigmoid())
    layer.add(node=6, output_num=2, output_func=ActivationMethods.softmax)
    y = layer.predict(x)
    print(y)
    # print(layer.hidden_layer)
    correct_count = 0
    counter = 0
    # for _ in range(100):
    #     x = np.array([1, 1, 1, 0, 1, 0])
    #     x = layer1.forward(x)
    #     x = layer2.forward(x)
    #     x = layer3.forward(x)
    #     loss_f = lambda W: LossFunctions.cross_entropy_error(x, np.array([1.0, 0.0]))
    #     layer1.weight = layer1.gradient_descent(f=loss_f, init_x=layer1.weight, lr=0.3)
    #     layer1.bias = layer1.gradient_descent(f=loss_f, init_x=layer1.bias, lr=0.3)
    #     layer2.weight = layer2.gradient_descent(f=loss_f, init_x=layer2.weight, lr=0.3)
    #     layer2.bias = layer2.gradient_descent(f=loss_f, init_x=layer2.bias, lr=0.3)
    #     layer3.weight = layer3.gradient_descent(f=loss_f, init_x=layer3.weight, lr=0.3)
    #     layer3.bias = layer3.gradient_descent(f=loss_f, init_x=layer3.bias, lr=0.3)

    # print(x)
    # print(LossFunctions.cross_entropy_error(x, [1, 0]))
    # print(LossFunctions.mean_squared_error(x, [1, 0]))
    # x = ActivationMethods.softmax(x)
    # print(x)
    # print(layer1.forward(np.array([1, 1])))
    # print(layer2.forward(np.array([1, 1])))
