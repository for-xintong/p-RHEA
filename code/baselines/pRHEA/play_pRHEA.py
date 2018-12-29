import gym, os, cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from baselines.pRHEA.utils import discount_with_dones
from baselines.pRHEA.run_pRHEA import Model, Environment
from baselines.pRHEA.policies import MlpPolicy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Runner(object):

    def __init__(self, envir, model, nenvs, nsteps, ngens, ninit, gamma=0.99):
        self.envir = envir
        self.model = model
        self.nenvs = nenvs
        self.nsteps = nsteps
        self.ngens = ngens
        self.ninit = ninit
        self.gamma = gamma

    def inital(self):
        global env_id, state_id, main_qpos, main_qvel, main_state, bestXsaved, bestXflags
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
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        return mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_scores, mb_states, mb_qpos, mb_qvel

    def evolution(self, x):
        global env_id, state_id, main_qpos, main_qvel, main_state
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
            actions = x[:, n*skip:(n+1)*skip]
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
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        return mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_scores, mb_states, mb_qpos, mb_qvel

    def CMA_run(self):
        global env_id, main_env, main_qpos, main_qvel, main_state, timesteps_sofar, t, score, bestXsaved, bestXflags
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
                best_rewards = mb_rewards[parents[0], :]
                best_actions = mb_actions[parents[0], :, :]
                best_dones = mb_dones[parents[0], :]
                best_scores = mb_scores[parents[0], :]

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

        T = 1
        bestXflags = bestXflags * 0
        for i in range(T, self.nsteps):
            bestXsaved[T-1, :] = best_actions[i, :]
            bestXflags[T-1] = 1
            if best_dones[i]:
                break

        return best_actions[0:T, :], best_scores[0:T], best_dones[0:T]


game_names = ['Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'InvertedPendulum-v2',
              'InvertedDoublePendulum-v2', 'Swimmer-v2', 'Walker2d-v2']
for game_name in ['Hopper-v2']:
    for model_num in range(1):
        for repeat in range(1):
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
            bestXsaved = np.zeros((nsteps, ac_space.shape[0]))
            bestXflags = np.zeros(nsteps)

            policy_fn = MlpPolicy
            model = Model(policy_fn, ob_space, ac_space, nenvs=nenvs, nsteps=nsteps, ninit=ninit, nbatch=0, lrschedule='constant')
            load_path = os.path.abspath('.') + "/result/%s_model_pRHEA_%d.pkl" % (game_name, model_num)
            model.load(load_path)

            env_id = []
            state_id = []
            for _ in range(nenvs):
                env = gym.make(game_name)
                obs = env.reset()
                env_id.append(env)
                state_id.append(obs)
            state_id = np.array(state_id)

            envir = Environment()
            runner = Runner(envir, model, nenvs=nenvs, nsteps=nsteps, ngens=ngens, ninit=ninit)
            score = 0
            steps = 0
            scores_step = [0]
            scores_cum = [0]

            while True:
                mb_actions, mb_scores, mb_dones = runner.CMA_run()
                for mb_action, mb_score, mb_done in zip(mb_actions, mb_scores, mb_dones):
                    # print([mb_score, mb_done])
                    img = main_env.render(mode='rgb_array')
                    im = Image.fromarray(img)
                    im.save(os.path.abspath('.') + "/result/%s/pRHEA/frame_%d.png" % (game_name, steps))
                    obs, reward, done, _= main_env.step(mb_action)
                    score += reward
                    scores_step.append(reward * 10)
                    scores_cum.append(score)
                    steps += 1
                    print([reward, done])
                    print([score, steps])
                    if done:
                        break

                img = main_env.render(mode='rgb_array')
                im = Image.fromarray(img)
                im.save(os.path.abspath('.') + "/result/%s/pRHEA/frame_%d.png" % (game_name, steps))
                main_qpos = main_env.env.data.qpos
                main_qvel = main_env.env.data.qvel
                main_state = obs

                if done:
                    break

            np.savetxt(os.path.abspath('.') + "/result/%s/pRHEA/single_step_reward.csv" % game_name, np.array(scores_step))
            plt.style.use('ggplot')
            plt.figure(dpi=400, figsize=(8, 3))
            plt.plot(range(steps + 1), scores_cum, 'r', linewidth="2")
            plt.plot(range(steps + 1), scores_step, 'b', linewidth="2")
            plt.legend(labels=["cumulative reward", "single step reward"], loc='best')
            plt.title('p-RHEA, H=20')
            plt.grid(True, linestyle="-", color="w", linewidth="1")
            plt.savefig(os.path.abspath('.') + '/result/%s/pRHEA/pRHEA_%s.png' % (game_name, game_name), format='png')

            img_root = os.path.abspath('.') + "/result/%s/pRHEA/" % game_name
            fps = 10
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            videoWriter = cv2.VideoWriter(os.path.abspath('.') + "/result/%s/pRHEA/saveVideo.avi" % game_name, fourcc, fps, img.shape[::-1][1:3])
            for i in range(steps + 1):
                frame = cv2.imread(img_root + 'frame_' + str(i) + '.png')
                videoWriter.write(frame)
            videoWriter.release()
            tf.reset_default_graph()
