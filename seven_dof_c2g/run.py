import argparse
import pdb
import seven_dof_c2g.config as config
from seven_dof_c2g.c2g_hof import Cost_HOF

parser = argparse.ArgumentParser()
parser = config.parse_arguments(parser)
args = parser.parse_args()

if args.test:
    assert args.pt_env_name is not None
    hof = Cost_HOF(args)
    hof.test()
elif args.keep_training:
    assert args.pt_env_name is not None
    print(args)
    hof = Cost_HOF(args)
    print(args)
    hof.keep_training()
else:
    hof = Cost_HOF(args)
    hof.train()
