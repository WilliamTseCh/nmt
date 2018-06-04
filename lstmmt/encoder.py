import numpy as np
import numpy as np
import csv

UNKNOWN = 'UNKNOWN'
START = 'START'
END = 'END'

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
    def __init__(self, data_dim, hidden_dim=100):
        # data_dim: 词向量维度，即词典长度; hidden_dim: 隐单元维度
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # 初始化权重向量
        self.whi, self.wxi, self.bi = self._init_wh_wx()
        self.whf, self.wxf, self.bf = self._init_wh_wx()
        self.who, self.wxo, self.bo = self._init_wh_wx()
        self.wha, self.wxa, self.ba = self._init_wh_wx()
        self.wy, self.by = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.data_dim, self.hidden_dim)), \
                           np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), (self.data_dim, 1))
        print ("whf:",self.whf)
        print ("wxf:",self.wxf)
        print ("bf:",self.bf)

        print ("whi:",self.whi)
        print ("wxi:",self.wxi)
        print ("bi:",self.bi)

        print ("wha:",self.wha)
        print ("wxa:",self.wxa)
        print ("ba:",self.ba)

        print ("who:",self.who)
        print ("wxo:",self.wxo)
        print ("bo:",self.bo)

        print ("wy:",self.wy)
        print ("by:",self.by)
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
        print ("whf:",self.whf)
        print ("wxf:",self.wxf)
        print ("bf:",self.bf)

        print ("whi:",self.whi)
        print ("wxi:",self.wxi)
        print ("bi:",self.bi)

        print ("wha:",self.wha)
        print ("wxa:",self.wxa)
        print ("ba:",self.ba)

        print ("who:",self.who)
        print ("wxo:",self.wxo)
        print ("bo:",self.bo)

        print ("wy:",self.wy)
        print ("by:",self.by)




    # 初始化 wh, wx, b
    def _init_wh_wx(self):
        wh = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim),(self.hidden_dim, self.hidden_dim))
        wx = np.random.uniform(-np.sqrt(1.0/self.data_dim), np.sqrt(1.0/self.data_dim),(self.hidden_dim, self.data_dim))
        b = np.random.uniform(-np.sqrt(1.0/self.data_dim), np.sqrt(1.0/self.data_dim),(self.hidden_dim, 1))

        return wh, wx, b

    # 初始化各个状态向量
    def _init_s(self, T):
        fss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # forget gate
        iss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # input gate
        ass = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # current inputstate
        oss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # output gate
        css = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # cell state
        hss = np.array([np.zeros((self.hidden_dim, 1))] * (T + 1))  # hidden state
        ys = np.array([np.zeros((self.data_dim, 1))] * T)    # output value


        return {'iss': iss, 'fss': fss, 'oss': oss,'ass': ass, 'hss': hss, 'css': css,'ys': ys}

    # 前向传播，单个x
    def forward(self, x):
        print ("=====================================forward(x)")
        print("向量时间长度")
        T = len(x)
        print ("T=",T)
        print("初始化各个状态向量")
        stats = self._init_s(T)
        for t in range(T):
            print ("t:",t)
            print ("stats:",stats)
            # 前一时刻隐藏状态
            # print("stats['hss'][t-3]:",stats['hss'][t-3])
            # print("stats['hss'][t-2]:",stats['hss'][t-2])
            print("stats['hss'][t-1]:",stats['hss'][t-1])
            print ("stats['hss']:",stats['hss'])
            # print("stats['hss'][t]:",stats['hss'][t])
            # print("stats['hss'][t+1]:",stats['hss'][t+1])
            # print("stats['hss'][t+2]:",stats['hss'][t+2])
            # print("stats['hss'][t+3]:",stats['hss'][t+3])


            ht_pre = np.array(stats['hss'][t-1]).reshape(-1, 1)
            print ("================================forget gate _cal_gate 开始:")
            stats['fss'][t] = self._cal_gate(self.whf, self.wxf, self.bf, ht_pre, x[t], sigmoid)
            print("forget gate _cal_gate 计算后:stats['fss'][t]",stats['fss'][t])
            print ("================================input gate _cal_gate 开始:")
            stats['iss'][t] = self._cal_gate(self.whi, self.wxi, self.bi, ht_pre, x[t], sigmoid)
            print ("input gate _cal_gate计算后 stats['iss'][t]:",stats['iss'][t])
            print ("================================current inputstate _cal_gate 开始:")
            stats['ass'][t] = self._cal_gate(self.wha, self.wxa, self.ba, ht_pre, x[t], tanh)
            print ("current inputstate计算后  stats['ass'][t]:",stats['ass'][t])
            print ("================================output gate _cal_gate开始:")
            stats['oss'][t] = self._cal_gate(self.who, self.wxo, self.bo, ht_pre, x[t], sigmoid)
            print ("output gate _cal_gate计算后 stats['oss'][t]:",stats['oss'][t])
            print ("cell state, ct = ft * ct_pre + it * at")
            print ("stats['fss'][t]:",stats['fss'][t])
            print("stats['css'][t-1]:",stats['css'][t-1])
            print ("stats['iss'][t]:",stats['iss'][t])
            print ("stats['ass'][t]:",stats['ass'][t])
            print ("stats['fss'][t] * stats['css'][t-1]:",stats['fss'][t] * stats['css'][t-1])
            print ("stats['iss'][t] * stats['ass'][t]:",stats['iss'][t] * stats['ass'][t])
            stats['css'][t] = stats['fss'][t] * stats['css'][t-1] + stats['iss'][t] * stats['ass'][t]
            print ("计算后stats['css'][t]:",stats['css'][t])
            print ("hidden state, ht = ot * tanh(ct)")
            print ("tanh(stats['css'][t]):",tanh(stats['css'][t]))
            print("stats['oss'][t]:",stats['oss'][t])
            stats['hss'][t] = stats['oss'][t] * tanh(stats['css'][t])
            print("计算后stats['hss'][t]:",stats['hss'][t])
            print ("output value, yt = softmax(self.wy.dot(ht) + self.by)")
            print ("self.wy.dot(stats['hss'][t]):",self.wy.dot(stats['hss'][t]))
            print ("self.wy.dot(stats['hss'][t]) + self.by:",self.wy.dot(stats['hss'][t]) + self.by)
            stats['ys'][t] = softmax(self.wy.dot(stats['hss'][t]) + self.by)
            print ("计算后stats['ys'][t]:",stats['ys'][t])
        return stats

    # 计算各个门的输出
    def _cal_gate(self, wh, wx, b, ht_pre, x, activation):
        print ("wh:",wh)
        print ("wx:",wx)
        print ("b:",b)
        print ("ht_pre:",ht_pre)
        print ("x:",x)
        print ("wh.dot(ht_pre):",wh.dot(ht_pre))
        print ("wx[:, x]:",wx[:, x])
        print ("wx[:, x].reshape(-1,1):",wx[:, x].reshape(-1,1))
        add=wh.dot(ht_pre) + wx[:, x].reshape(-1,1) + b
        print ("add:",add)
        return activation(wh.dot(ht_pre) + wx[:, x].reshape(-1,1) + b)

    # 预测输出，单个x
    def predict(self, x):
        stats = self.forward(x)
        pre_y = np.argmax(stats['ys'].reshape(len(x), -1), axis=1)
        return pre_y

    # 计算损失， softmax交叉熵损失函数， (x,y)为多个样本
    def loss(self, x, y):
        print ("loss计算损失开始:")
        cost = 0
        for i in range(len(y)):
            stats = self.forward(x[i])

            # 取出 y[i] 中每一时刻对应的预测值
            pre_yi = stats['ys'][range(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_yi))
        print ("stats:",stats)
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
        print ("stats:",stats)
        delta_o = stats['ys']
        print ("目标函数对输出 y 的偏导数delta_o = stats['ys']",delta_o)
        delta_o[np.arange(len(y)), y] -= 1
        print ("整个句子每个词语forward结束")
        print ("np.arange(len(y))[::-1]:",np.arange(len(y))[::-1])
        for t in np.arange(len(y))[::-1]:
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
            dwhf, dwxf, dbf = self._cal_grad_delta(dwhf, dwxf, dbf, delta_ft_net, stats['hss'][t-1], x[t])
            dwhi, dwxi, dbi = self._cal_grad_delta(dwhi, dwxi, dbi, delta_it_net, stats['hss'][t-1], x[t])
            dwha, dwxa, dba = self._cal_grad_delta(dwha, dwxa, dba, delta_at_net, stats['hss'][t-1], x[t])
            dwho, dwxo, dbo = self._cal_grad_delta(dwho, dwxo, dbo, delta_ot_net, stats['hss'][t-1], x[t])

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
        print ("样本(x,y,learning_rate):",(x,y,learning_rate))

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
    def train(self, X_train, y_train, learning_rate=0.005, n_epoch=1):
        losses = []
        num_examples = 0

        for epoch in range(n_epoch):
            for i in range(len(y_train)):
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples += 1

            loss = self.loss(X_train, y_train)
            losses.append(loss)
            print ("losses:")
            print ('epoch {0}: loss = {1}'.format(epoch+1, loss))
            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate *= 0.5
                print ('decrease learning_rate to', learning_rate)


 # 将文本拆成句子，并加上句子开始和结束标志
def _get_sentences(file_path):
        sents = []
        with open(file_path, 'rt',encoding="utf-8") as f:
            reader = csv.reader(f, skipinitialspace=True)
            sents = [row for row in reader]
            print ('Get {} sentences.'.format(len(sents)))
            return sents

    # 得到每句话的单词，并得到字典及字典中每个词的下标
def _get_dict_wordsIndex(sents,dict_size):
        sent_words_all=[]
        sent_words=[]
        word_freq={}
        sents_se=[]
        for sent in sents:
            str_sent=sent[0]
            sent_se=[]
            sent_se.append(START)
            sent_word_arr=str_sent.split(" ")
            for w in sent_word_arr:
                sent_se.append(w)
            sent_se.append(END)
            sents_se.append(sent_se)
        for sent in sents_se:
            for w in sent:
                if w!=START and w!=END:
                    sent_words_all.append(w)
        for w in sent_words_all:
            if w in word_freq:
                word_freq[w]+=1
            else:
                word_freq[w]=1
        word_freq[START]=10000
        word_freq[END]=0
        print ('Get {} words.'.format(len(word_freq)))

        common_words = sorted(word_freq, key=word_freq.get, reverse=True)
        common_words.append(UNKNOWN)
        dict_words = dict((word,ix) for ix,word in enumerate(common_words))
        index_of_words=dict((ix,word) for ix,word in enumerate(common_words))
        print ("sents_se:",sents_se)
        print ("dict_words:",dict_words)
        print ("index_of_words:",index_of_words)
        return sents_se, dict_words, index_of_words

    # 得到训练数据
def get_vector(file_path, dict_size):
        sents = _get_sentences(file_path)
        sent_words, dict_words, index_of_words = _get_dict_wordsIndex(sents,dict_size)
        print ("sent_words:",sent_words)
        # 将每个句子中没包含进词典dict_words中的词替换为unknown_token
        for i, words in enumerate(sent_words):
            sent_words[i] = [w if w in dict_words else UNKNOWN for w in words]
        print ("将每个句子中没包含进词典dict_words中的词替换为unknown_token 后sent_words:",sent_words)
        X_train = np.array([[dict_words[w] for w in sent] for sent in sent_words])
        y_train = np.array([[dict_words[w] for w in sent] for sent in sent_words])

        return X_train, y_train, dict_words, index_of_words

# def generate_text(rnn, dict_words, index_of_words):
#     # dict_words: type list; index_of_words: type dict
#     sent = [dict_words[START]]
#     # 预测新词，知道句子的结束(END_TOKEN)
#     while not sent[-1] == dict_words[END]:
#         next_probs, _ = rnn.forward(sent)
#         sample_word = index_of_words[UNKNOWN]
#         # 按预测输出分布进行采样，得到新的词
#         while sample_word == index_of_words[UNKNOWN]:
#             samples = np.random.multinomial(1, next_probs[-1])
#             sample_word = np.argmax(samples)
#         # 将新生成的有含义的词(即不为UNKNOWN_TOKEN的词)加入句子
#         sent.append(sample_word)
#     new_sent = [dict_words[i] for i in sent[1:-1]]
#     new_sent_str = ' '.join(new_sent)
#     return new_sent_str

# 获取数据
file_path = r'comment.csv'
dict_size = 3
X_train, y_train, dict_words, index_of_words = get_vector(file_path,dict_size)
lstm = myLSTM(dict_size, hidden_dim=2)
lstm.train(X_train[:200], y_train[:200],learning_rate=0.005,n_epoch=1)


print("训练结束,开始预测")
sent="你"
print ("你")
sent_word_arr=sent.split(" ")
sent_arr=[]
for i in sent_word_arr:
    sent_arr.append(index_of_words[i])
print (sent_arr)
prd=lstm.predict(sent_arr)
print(prd)

prd_words=[]
for i in prd:
    if i==index_of_words[data_utils.END]:
        break
    prd_words.append(index_of_words[i])
print ("prd_words:",prd_words)