import numpy as np;
import sys;
sys.path.append('../NeuralNetwork');
from NN import NN as NN;

# fetch data
xy_lst = [];

dirc = "./data/";
while True:
    try:
        filename = input(" data file to read: ");
        print(dirc+filename);
        f = open(dirc+filename);
        for line in f:
            words = line.split();
            xy_lst.append(list(map(float, words)));
        f.close();
    except IOError:
        print(IOError);
        break;

xy = np.array(xy_lst);
np.random.shuffle(xy);

x = xy.transpose()[:-1,:];
y = xy.transpose()[[-1],:];

print(" x shape: ", x.shape);
print(" y shape: ", y.shape);

# train tuning
m = y.shape[1];
m_train = int(0.7*m);
x_train = x[:,0:m_train];
y_train = y[:,0:m_train];
x_val = x[:,m_train:];
y_val = y[:,m_train:];

Layers = (17, 8, 8, 4, 1);
maxIteration = 5000;
alpha = 0.03;
reg = 0.0;
batch_size = 200;

nn = NN(Layers, alpha, reg);
nn.load('config');

n_batch = m_train // batch_size;
for epoch in range(0, maxIteration):
    for i in range(0, n_batch):
        start = i* batch_size;
        end = np.minimum(start+batch_size, m_train);
        nn.train_once(x_train[:,start:end], y_train[:,start:end]);
    print(epoch, " train cost J: ", np.asscalar(nn.J(nn.predict(x), y)));
    print(epoch, " test cost J: ", np.asscalar(nn.J(nn.predict(x_val), y_val)));

nn.test(x_val, y_val);
print(" accuracy: ", nn.accuracy);
print(" precision: ", nn.precision);
print(" recall: ", nn.recall);
print(" fScore: ", nn.fscore);

nn.save('config');
#nn.load('config');
