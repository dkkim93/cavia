import gym
import torch
import multiprocessing as mp
from envs.subproc_vec_env import SubprocVecEnv
from episode import BatchEpisodes


def make_env(env_name, args):
    def _make_env():
        env = gym.make(env_name)
        env._max_episode_steps = args.ep_horizon
        return env
    return _make_env


class BatchSampler(object):
    def __init__(self, args, log):
        self.args = args
        self.log = log

        self.queue = mp.Queue()
        self.envs = dict()
        for env_name in args.env_name:
            self.envs[env_name] = SubprocVecEnv(
                [make_env(env_name, args) for _ in range(args.num_workers)], queue=self.queue)
            self.envs[env_name].seed(args.seed)

        self._env = dict()
        for env_name in args.env_name:
            self._env[env_name] = gym.make(env_name)
            self._env[env_name]._max_episode_steps = args.ep_horizon
            self._env[env_name].seed(args.seed)

        self.test_env = dict()
        for env_name in args.env_name:
            self.test_env[env_name] = gym.make(env_name)
            self.test_env[env_name]._max_episode_steps = args.ep_horizon
            self.test_env[env_name].seed(args.seed)
        self.sample_test_tasks(num_tasks=1)

        self.observation_space = self.envs[args.env_name[0]].observation_space
        self.action_space = self.envs[args.env_name[0]].action_space

    def sample(self, policy, env_name, params=None, gamma=0.95, batch_size=None):
        if batch_size is None:
            batch_size = self.args.fast_batch_size

        episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.args.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(self.args.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs[env_name].reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.args.device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs[env_name].step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

        return episodes

    def reset_task(self, env_name, task):
        tasks = [task for _ in range(self.args.num_workers)]
        reset = self.envs[env_name].reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = dict()
        for env_name in self.args.env_name:
            tasks[env_name] = self._env[env_name].unwrapped.sample_tasks(num_tasks)
        return tasks

    def sample_test_tasks(self, num_tasks=1):
        self.test_tasks = dict()
        for env_name in self.args.env_name:
            self.test_tasks[env_name] = self._env[env_name].unwrapped.sample_tasks(num_tasks)
            self.log[self.args.log_name].info("[env::{}] Debug test: {}".format(
                env_name, self.test_tasks[env_name][0]))
