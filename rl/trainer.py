import torch
import numpy as np
import scipy.stats as st


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


def total_rewards(episodes_per_task, interval=False):
    returns = get_returns(episodes_per_task).cpu().numpy()

    mean = np.mean(returns, axis=0)
    conf_int = st.t.interval(0.95, len(mean) - 1, loc=mean, scale=st.sem(returns, axis=0))
    conf_int = mean - conf_int
    if interval:
        return mean, conf_int[0]
    else:
        return mean


def train(sampler, metalearner, args, log, tb_writer):
    for batch in range(args.num_batches):
        # Get a batch of tasks
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)

        # do the inner-loop update for each task
        # this returns training (before update) and validation (after update) episodes
        episodes, inner_losses = metalearner.sample(tasks, first_order=args.first_order)

        # take the meta-gradient step
        outer_loss = metalearner.step(
            episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)

        # For logging
        curr_returns = total_rewards(episodes, interval=True)
        log[args.log_name].info("Return after update: {}".format(curr_returns[0][1]))

        tb_writer.add_scalar('policy/actions_train', episodes[0][0].actions.mean(), batch)
        tb_writer.add_scalar('policy/actions_test', episodes[0][1].actions.mean(), batch)

        tb_writer.add_scalar('running_returns/before_update', curr_returns[0][0], batch)
        tb_writer.add_scalar('running_returns/after_update', curr_returns[0][1], batch)

        tb_writer.add_scalar('running_cfis/before_update', curr_returns[1][0], batch)
        tb_writer.add_scalar('running_cfis/after_update', curr_returns[1][1], batch)

        tb_writer.add_scalar('loss/inner_rl', np.mean(inner_losses), batch)
        tb_writer.add_scalar('loss/outer_rl', outer_loss.item(), batch)

        # Evaluation
        if batch % args.test_freq == 0:
            test_tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
            test_episodes = metalearner.test(test_tasks, num_steps=args.num_test_steps,
                                             batch_size=args.test_batch_size, halve_lr=args.halve_test_lr)
            all_returns = total_rewards(test_episodes, interval=True)
            for num in range(args.num_test_steps + 1):
                tb_writer.add_scalar('evaluation_rew/avg_rew ' + str(num), all_returns[0][num], batch)
                tb_writer.add_scalar('evaluation_cfi/avg_rew ' + str(num), all_returns[1][num], batch)

            log[args.log_name].info("Inner RL loss:: {}".format(np.mean(inner_losses)))
            log[args.log_name].info("Outer RL loss:: {}".format(outer_loss.item()))
