#!/usr/bin/env python3
import time, gym, joblib
import os, csv
import os.path as osp
from collections import deque
import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.common.cmd_util import atari_arg_parser
from baselines.common import tf_util

from baselines.pRHEA.policies import CnnPolicy, LstmPolicy, MlpPolicy
from baselines.pRHEA.replay_buffer import ReplayBuffer
from baselines.pRHEA.utils import discount_with_dones
from baselines.pRHEA.utils import Scheduler, make_path, find_trainable_variables
from baselines.pRHEA.utils import cat_neglogpac, cat_entropy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

env_id = []
state_id = []
main_qpos = None
main_qvel = None
main_state = None
timesteps_sofar = 0
t = 0
score = 0
bestXsaved = None
bestXflags = None

class Environment(object):

    def init(self, envs, nenvs):
        self.envs = envs
        self.nenvs = nenvs
        self.flags = np.zeros(nenvs)
        self.frames = []
        for id in range(nenvs):
            Q = deque([], maxlen=3)
            self.frames.append(Q)
        self.rewards = [0] * nenvs
        self.dones = [0] * nenvs
        self.zero_obs = np.zeros(envs[0].observation_space.shape)
        self.zero_qpos = np.zeros(envs[0].env.data.qpos.shape)
        self.zero_qvel = np.zeros(envs[0].env.data.qvel.shape)

    def func(self, inx, actions):
        global timesteps_sofar
        if self.flags[inx] == 0:
            obs, reward, done, _ = self.envs[inx].step(actions[inx,:])
            qpos = self.envs[inx].env.data.qpos
            qvel = self.envs[inx].env.data.qvel
            timesteps_sofar += 1
            if done:
                self.envs[inx].reset()
                self.flags[inx] = 1
        else:
            obs, reward, done, qpos, qvel = self.zero_obs, 0, 1, self.zero_qpos, self.zero_qvel

        self.frames[inx].append(obs)
        self.frames[inx].append(qpos)
        self.frames[inx].append(qvel)
        self.rewards[inx] = reward
        self.dones[inx] = done

    def step(self, actions):
        for inx in range(self.nenvs):
            self.func(inx, actions)
        obs = []
        qpos = []
        qvel = []
        for inx in range(self.nenvs):
            obs.append(self.frames[inx][0])
            qpos.append(self.frames[inx][1])
            qvel.append(self.frames[inx][2])
        return np.array(obs), np.array(self.rewards), np.array(self.dones), np.array(qpos), np.array(qvel)

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, ninit, nbatch,
            ent_coef=0.0, vf_coef=1.0, max_grad_norm=0.5, lr=3e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(10e6), lrschedule='constant'):

        self.sess = tf_util.make_session()
        A = tf.placeholder(tf.float32, [nbatch, ac_space.shape[0]])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(self.sess, ob_space, ac_space, nenvs, 1, reuse=False)
        init_model = policy(self.sess, ob_space, ac_space, ninit, 1, reuse=True)
        train_model = policy(self.sess, ob_space, ac_space, nbatch, nsteps, reuse=True)

        neglogpac = cat_neglogpac(train_model.mean, train_model.logstd, A)
        pg_loss = tf.reduce_mean(neglogpac)
        vf_loss = 0.5 * tf.reduce_mean(tf.square(train_model.vf - R))
        entropy = tf.reduce_mean(cat_entropy(train_model.logstd))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        _ = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, rewards, actions, values):
            advs = rewards - values
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:lr}
            policy_loss, value_loss, policy_entropy, _ = self.sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = self.sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            self.sess.run(restores)

        self.step = step_model.step
        self.init_step = init_model.step
        self.value = step_model.value
        self.init_value = init_model.value
        self.train = train
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=self.sess)

class Runner(object):

    def __init__(self, envir, model, nenvs, nsteps, ngens, ninit, writer, t_start, total_timesteps, gamma=0.99):
        self.envir = envir
        self.model = model
        self.nenvs = nenvs
        self.nsteps = nsteps
        self.ngens = ngens
        self.ninit = ninit
        self.writer = writer
        self.t_start = t_start
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.max_step_allowed = 0

    def inital(self):
        global env_id, state_id, main_qpos, main_qvel, main_state, t, bestXsaved, bestXflags
        for inx in range(self.ninit):
            env_id[inx].env.set_state(main_qpos, main_qvel)
            state_id[inx, :] = main_state
        self.envir.init(env_id[0:self.ninit], self.ninit)

        states = state_id[0:self.ninit,:]
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_states, mb_qpos, mb_qvel = [],[],[],[],[],[],[],[]
        for n in range(self.nsteps):
            actions, values, _, _ = self.model.init_step(states)
            if bestXflags[n] == 1:
                actions[0,:] = bestXsaved[n:n+1, :]
            skip = env_id[0].action_space.shape[0]
            lb = env_id[0].action_space.low[0]
            ub = env_id[0].action_space.high[0]
            for i in range(self.ninit):
                for j in range(skip):
                    if actions[i, j] < lb: actions[i, j] = lb
                    if actions[i, j] > ub: actions[i, j] = ub
            mb_obs.append(np.copy(states))
            mb_actions.append(actions)
            mb_values.append(values)
            obs, rewards, dones, qpos, qvel = self.envir.step(actions)
            mb_states.append(np.copy(obs))
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            mb_qpos.append(qpos)
            mb_qvel.append(qvel)
            states = obs

        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0)
        mb_scores = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_rewards = np.copy(mb_scores)*0.1
        mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_states = np.asarray(mb_states, dtype=np.float32).swapaxes(1, 0)
        mb_qpos = np.asarray(mb_qpos, dtype=np.float32).swapaxes(1, 0)
        mb_qvel = np.asarray(mb_qvel, dtype=np.float32).swapaxes(1, 0)

        last_values = self.model.init_value(states).tolist()
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0 and t > 5000:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        return mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_scores, mb_states, mb_qpos, mb_qvel

    def evolution(self, x):
        global env_id, state_id, main_qpos, main_qvel, main_state, t
        for inx in range(self.nenvs):
            env_id[inx].env.set_state(main_qpos, main_qvel)
            state_id[inx, :] = main_state
        self.envir.init(env_id, self.nenvs)

        skip = env_id[0].action_space.shape[0]
        states = state_id
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_states, mb_qpos, mb_qvel = [], [], [], [], [], [], [], []
        for n in range(self.nsteps):
            _, values, _, _ = self.model.step(states)
            mb_obs.append(np.copy(states))
            actions = x[:, n*skip : (n+1)*skip]
            mb_actions.append(actions)
            mb_values.append(values)
            obs, rewards, dones, qpos, qvel = self.envir.step(actions)
            mb_states.append(np.copy(obs))
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            mb_qpos.append(qpos)
            mb_qvel.append(qvel)
            states = obs

        mb_obs = np.asarray(mb_obs, dtype=np.float32).swapaxes(1, 0)
        mb_scores = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_rewards = np.copy(mb_scores)*0.1
        mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_states = np.asarray(mb_states, dtype=np.float32).swapaxes(1, 0)
        mb_qpos = np.asarray(mb_qpos, dtype=np.float32).swapaxes(1, 0)
        mb_qvel = np.asarray(mb_qvel, dtype=np.float32).swapaxes(1, 0)

        last_values = self.model.value(states).tolist()
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0 and t > 5000:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        return mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_scores, mb_states, mb_qpos, mb_qvel

    def CMA_run(self, replay_buffer):
        global env_id, main_qpos, main_qvel, main_state, timesteps_sofar, t, score, bestXsaved, bestXflags
        skip = env_id[0].action_space.shape[0]
        Dim = self.nsteps*skip
        NP = self.nenvs
        mu = int(NP/2)
        lb = env_id[0].action_space.low[0]
        ub = env_id[0].action_space.high[0]
        xmean = np.zeros((1, Dim))
        sigma = 0.3 * (ub-lb)

        weights = np.log(mu+0.5) - np.log(np.linspace(1,mu,mu))
        weights = weights / sum(weights)
        mueff = np.square(sum(weights)) / sum(weights*weights)
        cc = (4+mueff/Dim) / (Dim+4+2*mueff/Dim)
        cs = (mueff+2) / (Dim+mueff+5)
        c1 = 2 / (np.square(Dim+1.3)+mueff)
        cmu = min(1-c1, 2*(mueff-2+1/mueff)/(np.square(Dim+2)+mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(Dim+1))-1) + cs

        pc = np.zeros((1, Dim))
        ps = np.zeros((1, Dim))
        B = np.eye(Dim)
        DD = np.ones(Dim)
        C = B.dot(np.diag(np.square(DD))).dot(B.T)
        invsqrtC = B.dot(np.diag(1/DD)).dot(B.T)
        chiD = np.sqrt(Dim) * (1-1/(4*Dim)+1/(21*np.square(Dim)))

        best_rewards = -float('inf')*np.ones(self.nsteps)
        x = np.zeros((NP, Dim))
        for gens in range(self.ngens):
            if gens == 0:
                mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_scores, mb_states, mb_qpos, mb_qvel = self.inital()
                for i in range(self.nsteps):
                    x[0:self.ninit, i*skip:(i+1)*skip] = mb_actions[0:self.ninit, i, :]
            else:
                x = np.tile(xmean, (NP, 1)) + sigma*np.random.randn(NP, Dim).dot(np.diag(DD)).dot(B.T)
                for i in range(NP):
                    for j in range(Dim):
                        if x[i,j] < lb: x[i,j] = lb
                        if x[i,j] > ub: x[i,j] = ub
                mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_scores, mb_states, mb_qpos, mb_qvel = self.evolution(x)
            parents = np.argsort(-mb_rewards[:,0])
            if mb_rewards[parents[0], 0] > best_rewards[0]:
                best_obs = mb_obs[parents[0], :, :]
                best_rewards = mb_rewards[parents[0], :]
                best_values = mb_values[parents[0], :]
                best_actions = mb_actions[parents[0], :, :]
                best_dones = mb_dones[parents[0], :]
                best_scores = mb_scores[parents[0], :]
                best_states = mb_states[parents[0], :, :]
                best_qpos = mb_qpos[parents[0], :, :]
                best_qvel = mb_qvel[parents[0], :, :]

            xold = xmean
            xmean = weights[None, :].dot(x[parents[0:mu],:])

            ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff)*(xmean-xold).dot(invsqrtC)/sigma
            hsig = sum(sum(ps*ps))/(1-pow((1-cs),2*gens+2))/Dim < 2+4/(Dim+1)
            pc = (1-cc)*pc + hsig*np.sqrt(cc*(2-cc)*mueff)*(xmean-xold)/sigma

            dif = (x[parents[0:mu],:]-np.tile(xold, (mu, 1)))/sigma
            C = (1-c1-cmu+(1-hsig)*c1*cc*(2-cc))*C + \
                c1*(pc.T).dot(pc) + cmu*(dif.T).dot(np.diag(weights)).dot(dif)

            sigma = sigma*np.exp((cs/damps)*(np.linalg.norm(ps)/chiD-1))
            C = np.triu(C) + np.triu(C,1).T
            DD, B = np.linalg.eig(C)
            DD = np.sqrt(np.real(DD))
            B = np.real(B)
            invsqrtC = B.dot(np.diag(1/DD)).dot(B.T)

        steps_for_train = max(1, int((self.nsteps - sum(best_dones))/2))
        bestXflags = bestXflags * 0
        for i in range(steps_for_train, self.nsteps):
            bestXsaved[i-steps_for_train, :] = best_actions[i, :]
            bestXflags[i-steps_for_train] = 1
            if best_dones[i]:
                break

        for i in range(steps_for_train):
            replay_buffer.add(best_obs[i,:], best_rewards[i], best_actions[i,:], best_values[i])
            score += best_scores[i]
            main_state = best_states[i, :]
            main_qpos = best_qpos[i, :]
            main_qvel = best_qvel[i, :]
            self.max_step_allowed += 1
            t += 1
            if best_dones[i] or self.max_step_allowed == 1000:
                self.max_step_allowed = 0
                main_state = env_id[0].reset()
                main_qpos = env_id[0].env.data.qpos
                main_qvel = env_id[0].env.data.qvel
                self.writer.writerow([timesteps_sofar, t, int(time.time() - self.t_start), score])
                print([timesteps_sofar, t, int(time.time() - self.t_start), score])
                score = 0
                break

def learn(model, runner, nbatch, replay_buffer, repeat_num):
    global timesteps_sofar, t
    t0 = time.time()
    timesteps_before = timesteps_sofar
    for _ in range(repeat_num):
        runner.CMA_run(replay_buffer)
        for _ in range(50):
            if replay_buffer.__len__() > 5000:
                obs, rewards, actions, values = replay_buffer.sample(nbatch)
                policy_loss, value_loss, policy_entropy = model.train(obs, rewards, actions, values)
            else:
                policy_loss, value_loss, policy_entropy = None, None, None

    nseconds = time.time() - t0
    fps = int((timesteps_sofar - timesteps_before) / nseconds)
    logger.record_tabular("algorithm", "pRHEA")
    logger.record_tabular("ML_loss", policy_loss)
    logger.record_tabular("fps", fps)
    logger.record_tabular("policy_entropy", policy_entropy)
    logger.record_tabular("timesteps_sofar", "%.2fM" % float(timesteps_sofar/1e6))
    logger.record_tabular("timesteps_for_train", t)
    logger.record_tabular("value_loss", value_loss)
    logger.dump_tabular()
    return model

def train(game_name, total_timesteps, load_path, policy, lrschedule):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'mlp':
        policy_fn = MlpPolicy

    global env_id, state_id, main_qpos, main_qvel, main_state, timesteps_sofar, t, score, bestXsaved, bestXflags
    main_env = gym.make(game_name)
    obs = main_env.reset()
    main_qpos = main_env.env.data.qpos
    main_qvel = main_env.env.data.qvel
    main_state = obs

    ac_space = main_env.action_space
    ob_space = main_env.observation_space
    nsteps = 20
    nenvs = 4 + int(3*np.log(nsteps*ac_space.shape[0]))
    ninit = nenvs
    ngens = 5
    nbatch = 32
    bestXsaved = np.zeros((nsteps, ac_space.shape[0]))
    bestXflags = np.zeros(nsteps)

    model = Model(policy_fn, ob_space, ac_space, nenvs, nsteps, ninit, nbatch, lrschedule=lrschedule)
    if 1 - (load_path == None):
        model.load(load_path)
    t_start = time.time()

    env_id = []
    state_id = []
    timesteps_sofar = 0
    t = 0
    score = 0
    for _ in range(nenvs):
        env = gym.make(game_name)
        obs = env.reset()
        env_id.append(env)
        state_id.append(obs)
    state_id = np.array(state_id)

    path = os.path.abspath('.')
    monitorfile = open(path + "/result/%s_rewards_pRHEA.csv" % game_name, 'w', newline='')
    writer = csv.writer(monitorfile)
    replay_buffer = ReplayBuffer(20000)
    envir = Environment()
    runner = Runner(envir, model, nenvs, nsteps, ngens, ninit, writer, t_start, total_timesteps)
    while timesteps_sofar < total_timesteps:
        model = learn(model, runner, nbatch, replay_buffer, repeat_num=50)
    model.save(path + "/result/%s_model_pRHEA.pkl" % game_name)
    monitorfile.close()
    tf.reset_default_graph()

def main(game_name):
    parser = atari_arg_parser()
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'mlp'], default='mlp')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    args = parser.parse_args()
    logger.configure()

    if game_name in ['Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Swimmer-v2']:
        args.num_timesteps = int(50e6)
    train(game_name=game_name, total_timesteps=args.num_timesteps, load_path=None, policy=args.policy, lrschedule=args.lrschedule)

if __name__ == '__main__':
    game_names = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'InvertedPendulum-v2',
                  'InvertedDoublePendulum-v2', 'Swimmer-v2', 'Walker2d-v2']
    for game_name in ['Hopper-v2']:
        main(game_name)
