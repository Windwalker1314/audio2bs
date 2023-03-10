from arguments import *
from runner import Runner


def main():
    args = get_common_args()
    args = get_train_args(args)
    runner = Runner(args)
    runner.run()

if __name__=="__main__":
    main()
    
