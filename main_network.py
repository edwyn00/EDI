from encoderDecoderNetwork import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="densely extract the video frames and optical flows")
    parser.add_argument('--lr',default=0.005,type=float,help='learning_rate')
    parser.add_argument('--steps',default=32,type=int)
    parser.add_argument('--epochs',default=5,type=int)
    parser.add_argument('--seq_var',default=1,type=int)
    parser.add_argument('--normalization',default=True,type=bool)
    parser.add_argument('--test_size',default=5,type=int)
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    args=parse_args()

    learning_rate = args.lr
    num_step_per_epoch = args.steps
    epochs = args.epochs
    seq_var = args.seq_var
    normal = args.normalization
    fields = None
    test_size = args.test_size
    main(learning_rate = learning_rate, num_step_per_epoch = num_step_per_epoch, epochs = epochs, seq_var=seq_var, fields=fields, normal=normal, test_size=test_size)
