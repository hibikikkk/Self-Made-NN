import numpy as np

from functions import ActivationMethods


class NeuralNetwork:

    def __init__(self, input_size, node, bias=None, weight=None, act_func=None, output_func=None):
        """
        input_size:int 入力の数
        node:int ニューロン数
        weight:numpy.array 設定する層の入力サイズとノード数によって多次元配列を入力する。入力がない場合ランダムで決定
        act_func:method or str 活性化関数の設定。設定がない場合活性化関数を使用しない
        """
        self.input_size = input_size
        self.node = node
        if weight:
            self.weight = weight
        else:
            self.weight = np.random.normal(loc=0, scale=0.1, size=(input_size, node))
        if bias:
            self.bias = bias
        else:
            self.bias = np.random.normal(loc=0, scale=0.5, size=node)

        self.act_func = act_func
        self.affine = ActivationMethods.Affine(self.weight, self.bias)
        self.output_func = output_func

    def forward(self, x, t=None):
        """
        前向き推論
        :param x:numpy.array 入力
        :return: 計算結果
        """
        if self.act_func:
            if type(self.act_func) == "str":
                pass
            else:
                return self.act_func.forward(self.affine.forward(x))

        elif self.output_func:
            return self.output_func.forward(self.affine.forward(x), t)

        else:
            return self.affine.forward(x)

    def backward(self, dx):
        if self.act_func:
            return self.affine.backward(self.act_func.backward(dx))
        if self.output_func:
            return self.affine.backward(self.output_func.backward(dx))

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
        self.layer_info = None
        self.output = None

    def get_layer_info(self):
        if self.hidden_layer:
            self.layer_info = {"layer_num": len(self.hidden_layer) - 1,
                               "node_info": [{"function": a["activation"].__class__.__name__ if a["activation"] else a[
                                   "output_func"].__class__.__name__ if a["output_func"] else "Identity",
                                              "node_num": a["node"]} for a in self.hidden_layer]}

        return self.layer_info

    def add(self, **kwargs):
        """隠れ層を追加するメソッド

        :param node:int 隠れ層のノード数設定
        :param out:int 出力数の設定
        :param activation:ActivationMethod 活性化関数の設定
        :param output_func:ActivationMethod 出力関数関数の設定
        :return:None
        """
        hidden_act = None
        output_func = None
        for key in kwargs:
            if key == "node":
                hidden_node = kwargs[key]

            if key == "activation":
                hidden_act = kwargs[key]

            if key == "output_func":
                output_func = kwargs[key]

        self.hidden_layer.append(
            {"node": hidden_node, "activation": hidden_act, "output_func": output_func,
             "net": None})

    def predict(self, x, t=None):
        """予測メソッド

        :param x:numpy.ndarray 入力値
        :return:numpy.ndarray 予測結果
        """
        for i, val in enumerate(self.hidden_layer):
            if not self.hidden_layer[i]["net"]:
                self.hidden_layer[i]["net"] = NeuralNetwork(input_size=x.shape[1] or 0,
                                                            node=val["node"], act_func=val["activation"],
                                                            output_func=val["output_func"])

            if t is not None and len(self.hidden_layer) - 1 == i:
                x = self.hidden_layer[i]["net"].forward(x, t)
            else:
                x = self.hidden_layer[i]["net"].forward(x)

        return x

    def fit(self, x, t, epoch_num=10, learning_rate=0.01):
        for _ in range(epoch_num):
            result = self.predict(x, t)
            print(result)

            dout = 1
            self.hidden_layer.reverse()
            for i, val in enumerate(self.hidden_layer):
                dout = self.hidden_layer[i]["net"].backward(dout)
                self.hidden_layer[i]["net"].weight -= learning_rate * self.hidden_layer[i]["net"].affine.dW
                self.hidden_layer[i]["net"].bias -= learning_rate * self.hidden_layer[i]["net"].affine.db
            self.hidden_layer.reverse()


def random_layer_set(layer: Layer):
    for _ in range(np.random.randint(100)):
        num = np.random.randint(1, 10)
        if num % 2 == 0:
            layer.add(node=num, activation=ActivationMethods.Sigmoid())
        elif num % 3 == 0:
            layer.add(node=num, activation=None)
        else:
            layer.add(node=num, activation=ActivationMethods.Relu())
    return layer


if __name__ == "__main__":
    # ファイルからデータセットの呼び出し
    import pandas as pd

    csv = pd.read_csv("data-set.csv")
    X = [[[csv["X1"][0]], [csv["X1"][1]]], [[csv["X2"][0]], [csv["X2"][1]]], [[csv["X3"][0]], [csv["X3"][1]]],
         [[csv["X4"][0]], [csv["X4"][1]]]]
    T = [csv["T1"], csv["T2"], csv["T3"], csv["T4"]]

    # print(np.array(X))
    # print(np.array(T))
    # x = np.array([[1], [1]])
    # t = np.array([1, 0])
    x = np.array(X)
    t = np.array(T)

    layer = Layer()

    # 隠れ層の追加
    layer.add(node=6, activation=ActivationMethods.Sigmoid())
    layer.add(node=6, activation=ActivationMethods.Relu())
    layer.add(node=6, activation=ActivationMethods.Sigmoid())
    layer.add(node=6, activation=ActivationMethods.Relu())

    # 最終層
    layer.add(node=2, output_func=ActivationMethods.SoftmaxWithLoss())
    layer.fit(x, t, epoch_num=100, learning_rate=0.01)
    # import pprint
    #
    # # pprint.pprint(layer.hidden_layer)
    # pprint.pprint(layer.get_layer_info())

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
