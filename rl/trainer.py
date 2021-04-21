import torch
import numpy as np


def get_returns(episodes_per_task):
    # sum up for each rollout, then take the mean across rollouts
    returns = []
    for task_idx in range(len(episodes_per_task)):
        curr_returns = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # compute returns for individual rollouts
            ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)
            curr_returns.append(ret)
        # result will be: num_evals * num_updates
        returns.append(torch.stack(curr_returns, dim=1))

    # result will be: num_tasks * num_evals * num_updates
    returns = torch.stack(returns)
    returns = returns.reshape((-1, returns.shape[-1]))

    return returns


def total_rewards(episodes_per_task, args, interval=False):
    returns = get_returns(episodes_per_task).cpu().numpy()
    returns = np.split(returns, len(args.env_name), axis=0)

    before_rewards, after_rewards = [], []
    for return_ in returns:
        mean = np.mean(return_, axis=0)
        before_rewards.append(mean[0])
        after_rewards.append(mean[1])
    return before_rewards, after_rewards


def train(sampler, metalearner, args, log, tb_writer):
    for batch in range(args.num_batches):
        # Get a batch of tasks
        assert args.meta_batch_size % len(args.env_name) == 0
        num_tasks = int(args.meta_batch_size / len(args.env_name))
        tasks = sampler.sample_tasks(num_tasks=num_tasks)

        # do the inner-loop update for each task
        # this returns training (before update) and validation (after update) episodes
        episodes, inner_losses = metalearner.sample(tasks, first_order=args.first_order)

        # take the meta-gradient step
        outer_loss = metalearner.step(
            episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)

        # For logging
        before_rewards, after_rewards = total_rewards(episodes, args, interval=True)
        for i_env, env in enumerate(args.env_name):
            log[args.log_name].info("[env::{}] Return before update: {} at {}".format(env, before_rewards[i_env], batch))
            tb_writer.add_scalars('running_returns/' + env, {"before": before_rewards[i_env]}, batch)

            log[args.log_name].info("[env::{}] Return after update: {} at {}".format(env, after_rewards[i_env], batch))
            tb_writer.add_scalars('running_returns/' + env, {"after": after_rewards[i_env]}, batch)

        tb_writer.add_scalar('policy/actions_train', episodes[0][0].actions.mean(), batch)
        tb_writer.add_scalar('policy/actions_test', episodes[0][1].actions.mean(), batch)

        tb_writer.add_scalar('loss/inner_rl', np.mean(inner_losses), batch)
        tb_writer.add_scalar('loss/outer_rl', outer_loss.item(), batch)

        # For debugging
        metalearner.sample_debug(sampler.test_tasks, batch, first_order=args.first_order)
