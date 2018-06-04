import numpy as np
from lstmmt import data_utils


# 输出单元激活函数
def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def logistic_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def tanh_deriv(x):
    return 1.0-np.tanh(x)*np.tanh(x)

class myLSTM:
    def __init__(self, encoder_dict_dim,decoder_dict_dim, hidden_dim=100,output_size=5):
        # data_dim: 词向量维度，即词典长度; hidden_dim: 隐单元维度
        self.encoder_dict_dim = encoder_dict_dim
        self.decoder_dict_dim = decoder_dict_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # 初始化权重向量
        self.whi, self.wxi, self.bi = self._init_wh_wx()
        self.whf, self.wxf, self.bf = self._init_wh_wx()
        self.who, self.wxo, self.bo = self._init_wh_wx()
        self.wha, self.wxa, self.ba = self._init_wh_wx()

        self.wy, self.by = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.decoder_dict_dim, self.hidden_dim)), \
                           np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.decoder_dict_dim, 1))
        self.whf[0][0]=0.11
        self.whf[0][1]=0.13
        self.whf[1][0]=0.12
        self.whf[1][1]=0.14

        self.wxf[0][0]=0.11
        self.wxf[0][1]=0.13
        self.wxf[0][2]=0.15
        self.wxf[1][0]=0.12
        self.wxf[1][1]=0.14
        self.wxf[1][2]=0.16

        self.bf[0][0]=0.11
        self.bf[1][0]=0.12

        self.whi[0][0]=0.21
        self.whi[0][1]=0.23
        self.whi[1][0]=0.22
        self.whi[1][1]=0.24

        self.wxi[0][0]=0.21
        self.wxi[0][1]=0.23
        self.wxi[0][2]=0.25
        self.wxi[1][0]=0.22
        self.wxi[1][1]=0.24
        self.wxi[1][2]=0.26

        self.bi[0][0]=0.21
        self.bi[1][0]=0.22

        self.wha[0][0]=0.31
        self.wha[0][1]=0.33
        self.wha[1][0]=0.32
        self.wha[1][1]=0.34

        self.wxa[0][0]=0.31
        self.wxa[0][1]=0.33
        self.wxa[0][2]=0.35
        self.wxa[1][0]=0.32
        self.wxa[1][1]=0.34
        self.wxa[1][2]=0.36

        self.ba[0][0]=0.31
        self.ba[1][0]=0.32


        self.who[0][0]=0.41
        self.who[0][1]=0.43
        self.who[1][0]=0.42
        self.who[1][1]=0.44

        self.wxo[0][0]=0.41
        self.wxo[0][1]=0.43
        self.wxo[0][2]=0.45
        self.wxo[1][0]=0.42
        self.wxo[1][1]=0.44
        self.wxo[1][2]=0.46

        self.bo[0][0]=0.41
        self.bo[1][0]=0.42

        self.wy[0][0]=0.51
        self.wy[0][1]=0.54
        self.wy[1][0]=0.52
        self.wy[1][1]=0.55
        self.wy[2][0]=0.53
        self.wy[2][1]=0.56

        self.by[0][0]=0.51
        self.by[1][0]=0.52
        self.by[2][0]=0.53
        print("self.whi, self.wxi, self.bi:")
        print(     self.whi)
        print(    self.wxi)
        print(    self.bi)
        print("self.whf, self.wxf, self.bf:")
        print(     self.whf)
        print(    self.wxf)
        print(    self.bf)
        print("self.wha, self.wxa, self.ba:")
        print(     self.wha)
        print(    self.wxa)
        print(    self.ba)
        print("self.who, self.wxo, self.bo:")
        print(     self.who)
        print(    self.wxo)
        print(    self.bo)
        print("self.wy, self.by:")
        print(self.wy)
        print(self.by)
    # 初始化 wh, wx, b
    def _init_wh_wx(self):
        wh = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),(self.hidden_dim, self.hidden_dim))
        wx = np.random.uniform(-np.sqrt(1.0/self.encoder_dict_dim), np.sqrt(1.0/self.encoder_dict_dim),(self.hidden_dim, self.encoder_dict_dim))
        b = np.random.uniform(-np.sqrt(1.0/self.encoder_dict_dim), np.sqrt(1.0/self.encoder_dict_dim),(self.hidden_dim, 1))

        return wh, wx, b

    # 初始化各个状态向量
    def _init_s(self, T):
        fss = np.array([np.zeros((self.hidden_dim, 1))] * T)  # forget gate
        iss = np.array([np.zeros((self.hidden_dim, 1))] * T)  # input gate
        ass = np.array([np.zeros((self.hidden_dim, 1))] * T)  # current inputstate
        oss = np.array([np.zeros((self.hidden_dim, 1))] * T)  # output gate
        css = np.array([np.zeros((self.hidden_dim, 1))] * T)  # cell state
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # hidden state
        ys = np.array([np.zeros((self.decoder_dict_dim, 1))] * T)    # output value

        return {'iss': iss, 'fss': fss, 'oss': oss,'ass': ass, 'hss': hss, 'css': css,'ys': ys}

    def forward(self, x):
       # print ("=====================================forward(x)")
        T =self.output_size
        #print ("T=",T)
        stats = self._init_s(T)
        for t in range(T):
            if t<len(x):
                #print ("t:",t)
                # 前一时刻隐藏状态
                ht_pre = np.array(stats['hss'][t-1]).reshape(-1, 1)
                # print("ht_pre:",ht_pre)
                # print("self.whf:",self.whf)
                # print("self.wxf:",self.wxf)
                # print("self.bf:",self.bf)
                # print("x:",x)
                stats['fss'][t] = self._cal_gate(self.whf, self.wxf, self.bf, ht_pre, x[t], sigmoid)
                stats['iss'][t] = self._cal_gate(self.whi, self.wxi, self.bi, ht_pre, x[t], sigmoid)
                stats['ass'][t] = self._cal_gate(self.wha, self.wxa, self.ba, ht_pre, x[t], tanh)
                stats['oss'][t] = self._cal_gate(self.who, self.wxo, self.bo, ht_pre, x[t], sigmoid)
                stats['css'][t] = stats['fss'][t] * stats['css'][t-1] + stats['iss'][t] * stats['ass'][t]
                stats['hss'][t] = stats['oss'][t] * tanh(stats['css'][t])
                stats['ys'][t] = softmax(self.wy.dot(stats['hss'][t]) + self.by)
        return stats

    # # 前向传播，单个x
    # def forward(self, x):
    #    # print ("=====================================forward(x)")
    #     T =self.output_size
    #     #print ("T=",T)
    #     stats = self._init_s(T)
    #     ht_pre = np.array(stats['hss'][-1]).reshape(-1, 1)
    #     print("ht_pre:",ht_pre)
    #     print("self.whf:",self.whf)
    #     print("self.wxf:",self.wxf)
    #     print("self.bf:",self.bf)
    #     print("x:",x)
    #     stats['fss'][0] = self._cal_gate(self.whf, self.wxf, self.bf, ht_pre, x, sigmoid)
    #     stats['iss'][0] = self._cal_gate(self.whi, self.wxi, self.bi, ht_pre, x, sigmoid)
    #     stats['ass'][0] = self._cal_gate(self.wha, self.wxa, self.ba, ht_pre, x, tanh)
    #     stats['oss'][0] = self._cal_gate(self.who, self.wxo, self.bo, ht_pre, x, sigmoid)
    #     stats['css'][0] = stats['fss'][0] * stats['css'][-1] + stats['iss'][0] * stats['ass'][0]
    #     stats['hss'][0] = stats['oss'][0] * tanh(stats['css'][0])
    #     stats['ys'][0] = softmax(self.wy.dot(stats['hss'][0]) + self.by)
    #     print("stats['ys'][t]:",stats['ys'][0])
    #     for t in range(1,T):
    #             #print ("t:",t)
    #             # 前一时刻隐藏状态
    #             ht_pre = np.array(stats['hss'][t-1]).reshape(-1, 1)
    #             y_pre = np.argmax(stats['ys'][t-1].reshape(1, -1), axis=1)
    #             print("ht_pre:",ht_pre)
    #             stats['fss'][t] = self._cal_gate(self.whf, self.wxf, self.bf, ht_pre, y_pre, sigmoid)
    #             stats['iss'][t] = self._cal_gate(self.whi, self.wxi, self.bi, ht_pre, y_pre, sigmoid)
    #             stats['ass'][t] = self._cal_gate(self.wha, self.wxa, self.ba, ht_pre, y_pre, tanh)
    #             stats['oss'][t] = self._cal_gate(self.who, self.wxo, self.bo, ht_pre, y_pre, sigmoid)
    #             stats['css'][t] = stats['fss'][t] * stats['css'][t-1] + stats['iss'][t] * stats['ass'][t]
    #             stats['hss'][t] = stats['oss'][t] * tanh(stats['css'][t])
    #             stats['ys'][t] = softmax(self.wy.dot(stats['hss'][t]) + self.by)
    #             print ("t:",t)
    #             print("stats['ys'][t]:",stats['ys'][t])
    #     return stats


    # 计算各个门的输出
    def _cal_gate(self, wh, wx, b, ht_pre, x, activation):
        a1=wh.dot(ht_pre)
        a2=wx[:, x].reshape(-1,1)

        add=wh.dot(ht_pre) + wx[:, x].reshape(-1,1) + b
        return activation(wh.dot(ht_pre) + wx[:, x].reshape(-1,1) + b)


    # 预测输出，单个x
    def predict(self, x):
        stats = self.forward(x)
        # print ("stats['ys']:",stats['ys'])
        preds=[]
        for i in range(self.output_size):
            pre_y = np.argmax(stats['ys'][i].reshape(1, -1), axis=1)
            # print ("pre_y:",pre_y[0])
            preds.append(pre_y[0])
        return preds

    # 计算损失， softmax交叉熵损失函数， (x,y)为多个样本
    def loss(self, x, y):
        print ("loss计算损失开始:")
        cost = 0
        for i in range(len(x)):
            # print("x[i]:",x[i])
            # print ("y[i]:",y[i])
            stats = self.forward(x[i])
            # print ("stats['ys']:",stats['ys'])
            # print ("y[i]:",y[i])
            pre_yi = stats['ys'][range(len(y[i])), y[i]] # 取出 y[i] 中每一时刻对应的预测值,还真的是
            # print("pre_yi:",pre_yi)

            count=len(pre_yi)
            for mm in range(len(pre_yi)):
                if pre_yi[mm][0]==0:
                    count=mm
                    break
            pre_yi1=np.random.uniform(-1, 1, (count, 1))
            for ii in range(count):
                pre_yi1[ii]=pre_yi[ii]
            cost -= np.sum(np.log(pre_yi1))
        # 统计所有y中词的个数, 计算平均损失
        N = np.sum([len(yi) for yi in y])
        ave_loss = cost / N
        return ave_loss

     # 初始化偏导数 dwh, dwx, db
    def _init_wh_wx_grad(self):
        dwh = np.zeros(self.whi.shape)
        dwx = np.zeros(self.wxi.shape)
        db = np.zeros(self.bi.shape)
        return dwh, dwx, db

    # 求梯度, (x,y)为一个样本
    def bptt(self, x, y):
        dwhi, dwxi, dbi = self._init_wh_wx_grad()
        dwhf, dwxf, dbf = self._init_wh_wx_grad()
        dwho, dwxo, dbo = self._init_wh_wx_grad()
        dwha, dwxa, dba = self._init_wh_wx_grad()
        dwy, dby = np.zeros(self.wy.shape), np.zeros(self.by.shape)

        # 初始化 delta_ct，因为后向传播过程中，此值需要累加
        delta_ct = np.zeros((self.hidden_dim, 1))

        # 前向计算
        stats = self.forward(x)
        # print ("stats:",stats)
        delta_o = stats['ys']

        # print ("delta_o:",delta_o)
        #print ("y:",[decoder_index_of_words[i] for i in y])
        # print ("目标函数对输出 y 的偏导数delta_o = stats['ys']",delta_o)
        # print ("np.arange(len(y)):",np.arange(len(y)))
        # print ("y:",y)
        delta_o[np.arange(len(y)), y] -= 1

        # print ("整个句子每个词语forward结束")
        # print ("np.arange(len(y))[::-1]:",np.arange(len(y))[::-1])
        for t in range(self.output_size):
            # print ("t:",t)
            # 输出层wy, by的偏导数，由于所有时刻的输出共享输出权值矩阵，故所有时刻累加
            dwy += delta_o[t].dot(stats['hss'][t].reshape(1, -1))
            dby += delta_o[t]

            # 目标函数对隐藏状态的偏导数
            delta_ht = self.wy.T.dot(delta_o[t])

            # 各个门及状态单元的偏导数
            delta_ct += delta_ht * stats['oss'][t] * (1-tanh(stats['css'][t])**2)
            delta_ot = delta_ht * tanh(stats['css'][t])
            delta_at = delta_ct * stats['iss'][t]
            delta_it = delta_ct * stats['ass'][t]
            delta_ft = delta_ct * stats['css'][t-1]

            delta_ft_net = delta_ft * stats['fss'][t] * (1-stats['fss'][t])
            delta_it_net = delta_it * stats['iss'][t] * (1-stats['iss'][t])
            delta_at_net = delta_at * (1-stats['ass'][t]**2)
            delta_ot_net = delta_ot * stats['oss'][t] * (1-stats['oss'][t])

            # 更新各权重矩阵的偏导数，由于所有时刻共享权值，故所有时刻累加
            if t<len(x):
                dwhf, dwxf, dbf = self._cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t-1], x[t])
                dwhi, dwxi, dbi = self._cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, stats['hss'][t-1], x[t])
                dwha, dwxa, dba = self._cal_grad_delta(dwha, dwxa, dba, delta_at_net, stats['hss'][t-1], x[t])
                dwho, dwxo, dbo = self._cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, stats['hss'][t-1], x[t])

            # if t==0:
            #     dwhf, dwxf, dbf = self._cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t-1], x)
            #     dwhi, dwxi, dbi = self._cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, stats['hss'][t-1], x)
            #     dwha, dwxa, dba = self._cal_grad_delta(dwha, dwxa, dba, delta_at_net, stats['hss'][t-1], x)
            #     dwho, dwxo, dbo = self._cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, stats['hss'][t-1], x)
            # else:
            #     y_pre = np.argmax(stats['ys'][t-1].reshape(1, -1), axis=1)
            #     dwhf, dwxf, dbf = self._cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t-1], y_pre)
            #     dwhi, dwxi, dbi = self._cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, stats['hss'][t-1], y_pre)
            #     dwha, dwxa, dba = self._cal_grad_delta(dwha, dwxa, dba, delta_at_net, stats['hss'][t-1], y_pre)
            #     dwho, dwxo, dbo = self._cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, stats['hss'][t-1], y_pre)

        return [dwhf, dwxf, dbf,
                dwhi, dwxi, dbi,
                dwha, dwxa, dba,
                dwho, dwxo, dbo,
                dwy, dby]

    # 更新各权重矩阵的偏导数
    def _cal_grad_delta(self, dwh, dwx, db, delta_net, ht_pre, x):
        dwh += delta_net * ht_pre
        dwx += delta_net * x
        db += delta_net

        return dwh, dwx, db

    # 计算梯度, (x,y)为一个样本
    def sgd_step(self, x, y, learning_rate):
       # print ("样本(x,y,learning_rate):",(x,y,learning_rate))

        dwhf, dwxf, dbf, \
        dwhi, dwxi, dbi, \
        dwha, dwxa, dba, \
        dwho, dwxo, dbo, \
        dwy, dby = self.bptt(x, y)

        # 更新权重矩阵
        self.whf, self.wxf, self.bf = self._update_wh_wx(learning_rate, self.whf, self.wxf, self.bf, dwhf, dwxf, dbf)
        self.whi, self.wxi, self.bi = self._update_wh_wx(learning_rate, self.whi, self.wxi, self.bi, dwhi, dwxi, dbi)
        self.wha, self.wxa, self.ba = self._update_wh_wx(learning_rate, self.wha, self.wxa, self.ba, dwha, dwxa, dba)
        self.who, self.wxo, self.bo = self._update_wh_wx(learning_rate, self.who, self.wxo, self.bo, dwho, dwxo, dbo)
        self.wy, self.by = self.wy - learning_rate * dwy, self.by - learning_rate * dby

    # 更新权重矩阵
    def _update_wh_wx(self, learning_rate, wh, wx, b, dwh, dwx, db):
        wh -= learning_rate * dwh
        wx -= learning_rate * dwx
        b -= learning_rate * db

        return wh, wx, b

     # 训练 LSTM
    def train(self, X_train, y_train, learning_rate=0.005, n_epoch=1000000):
        losses = []
        num_examples = 0
        for epoch in range(n_epoch):
            #print("迭代前self.whf, self.wxf, self.bf:",self.whf, self.wxf, self.bf)
            for i in range(len(y_train)):
                # print ("X_train[i], y_train[i]:",X_train[i], y_train[i])
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples += 1
            loss = self.loss(X_train, y_train)
            losses.append(loss)
           # print ("losses:")
            print ('epoch {0}: loss = {1}'.format(epoch+1, loss))
            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate *= 0.5
                print ('decrease learning_rate to', learning_rate)
           # print("迭代后self.whf, self.wxf, self.bf:",self.whf, self.wxf, self.bf)

