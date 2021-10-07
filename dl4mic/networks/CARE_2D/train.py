# ------------------ Diclaimer ------------------
#WIP> DON'T TRY TO USE THIS NETWORK YET
# -----------------------------------------------
import sys
import os
import time
import math
import warnings
import shutil
import csv
import tensorflow as tf
# import glob

sys.path.append(r'../lib/')
from dl4mic.cliparser import ParserCreator
from dl4mic.care import *

def main(argv):
    parser = ParserCreator.createArgumentParser("./train.yml")
    if len(argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args(argv[1:])
    print(args)

    if tf.test.gpu_device_name() == '':
        print('You do not have GPU access.')
        print('Expect slow performance.')
    else:
        print('You have GPU access')

    print('Tensorflow version is ' + str(tf.__version__))


    #Prepare Data - TODO

    #Create Model - TODO

    #Display Model and Training Parameters - OK
    model.keras_model.summary();

    # ------------------ Display ------------------
    print('---------------------------- Main training parameters ----------------------------')
    print('Number of epochs: '+str(args.epochs))
    print('Batch size: '+str(args.batchSize))
    print('Number of training dataset: '+str(len(X)))
    print('Number of training steps: '+str(number_of_steps-n_val))
    print('Number of validation steps: '+str(n_val))
    print('---------------------------- ------------------------ ----------------------------')

    start = time.time()
    #Train Model - TODO

    #history = model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augment,
    #                      epochs=args.epochs,
    #                      steps_per_epoch=number_of_steps)

    #Write to CSV - OK
    lossDataCSVPath = os.path.join(full_model_path, 'Quality Control/training_evaluation.csv')

    with open(lossDataCSVPath, 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.history.keys())
        values = list(history.history.values())
        #writer.writerows(values)
        for i in range(args.epochs):
            v=[values[j][i] for j in range(len(values))]
            writer.writerow(v)

    #Print End - OK
    print("------------------------------------------")
    dt = time.time() - start
    mins, sec = divmod(dt, 60)
    hour, mins = divmod(mins, 60)
    print("Time elapsed:", hour, "hour(s)", mins, "min(s)", round(sec), "sec(s)")
    print("------------------------------------------")

    print("---training done---")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main(sys.argv)
    