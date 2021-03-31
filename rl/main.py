import os
import numpy as np
from misc.utils import set_seed, set_log
from arguments import parse_args
from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.normal_mlp import CaviaMLPPolicy
from sampler import BatchSampler
from tensorboardX import SummaryWriter
from trainer import train


def main(args):
    # Set logging
    if not os.path.exists("./log"):
        os.makedirs("./log")

    log = set_log(args)
    tb_writer = SummaryWriter('./log/tb_{0}'.format(args.log_name))

    # Set seed
    set_seed(args.seed, cudnn=args.make_deterministic)

    # Set sampler
    sampler = BatchSampler(args, log)

    # Set policy
    policy = CaviaMLPPolicy(
        input_size=int(np.prod(sampler.observation_space.shape)),
        output_size=int(np.prod(sampler.action_space.shape)),
        hidden_sizes=(args.hidden_size,) * args.num_layers,
        num_context_params=args.num_context_params,
        device=args.device)

    # Initialise baseline
    baseline = LinearFeatureBaseline(int(np.prod(sampler.observation_space.shape)))

    # Initialise meta-learner
    metalearner = MetaLearner(sampler, policy, baseline, args, tb_writer)

    # Begin train
    train(sampler, metalearner, args, log, tb_writer)


if __name__ == '__main__':
    args = parse_args()
    main(args)
