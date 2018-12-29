import csv, os
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

game_names = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'InvertedPendulum-v2',
              'InvertedDoublePendulum-v2', 'Swimmer-v2', 'Walker2d-v2']
for game_name in game_names:
    path = os.path.abspath('.')
    skip = 2e5

    score = []
    length_X = []
    for run_num in range(5):
        monitorfile = open(path + "/result/%s_rewards_pRHEA_%d.csv" % (game_name, run_num), 'r')
        reader = csv.reader(monitorfile)
        Q = deque([], maxlen=20)
        Y = []
        flag = True
        num = 1
        for line in reader:
            Q.append(float(line[3]))
            if len(Q) == 5 and flag:
                Y.append(np.mean(Q))
                flag = False
            if int(line[0])>= num*skip:
                Y.append(np.mean(Q))
                num = num + 1
        Y.append(np.mean(Q))
        length_X.append(len(Y))
        X = np.arange(0, min(length_X), 1)*skip
        plt.plot(X,Y[0:min(length_X)])
        score.append(Y[0:min(length_X)])

    score = np.array(score)
    mean = np.mean(score, 0)
    std = np.std(score, 0)

    plt.style.use('ggplot')
    plt.figure(dpi=400, figsize=(4.2, 4))
    plt.plot(X, mean, 'r', linewidth = "2")
    plt.fill_between(X, mean+std, mean-std,color='r',alpha=0.25)
    plt.xlabel('Number of steps')
    plt.title(game_name)
    plt.grid(True, linestyle = "-", color = "w", linewidth = "1")
    plt.savefig(os.path.abspath('.') + '/result/%s.png' % game_name, format='png')
    plt.show()