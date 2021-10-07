import sys
import os
import numpy as np
from io import StringIO
sys.stderr = StringIO()
sys.path.append(r'../lib/')
from dl4mic.cliparser import ParserCreator
from dl4mic.unet import *
sys.stderr = sys.__stderr__

def main(argv):
    prediction_prefix = ""
    W = '\033[0m'  # white (normal)
    R = '\033[31m'  # red
    parser = ParserCreator.createArgumentParser("./predict.yml")
    args = parser.parse_args(argv[1:])
    full_Prediction_model_path = os.path.join(args.baseDir, args.name)
    if os.path.exists(os.path.join(full_Prediction_model_path, 'weights_best.hdf5')):
        print("The " + args.name + " network will be used.")
    else:
        print(R + '!! WARNING: The chosen model does not exist !!' + W)
        print('Please make sure you provide a valid model path and model name before proceeding further.')

    unet = load_model(os.path.join(args.baseDir, args.name, 'weights_best.hdf5'),
                      custom_objects={'_weighted_binary_crossentropy': weighted_binary_crossentropy(np.ones(2))})
    Input_size = unet.layers[0].output_shape[1:3]
    print('Model input size: ' + str(Input_size[0]) + 'x' + str(Input_size[1]))

    source_dir_list = os.listdir(args.dataPath)
    number_of_dataset = len(source_dir_list)
    print('Number of dataset found in the folder: ' + str(number_of_dataset))

    predictions = []
    for i in range(number_of_dataset):
        print("processing dataset " + str(i + 1) + ", file: " + source_dir_list[i])
        predictions.append(predict_as_tiles(os.path.join(args.dataPath, source_dir_list[i]), unet))

    # Save the results in the folder along with the masks according to the set threshold
    saveResult(args.output, predictions, source_dir_list, prefix=prediction_prefix, threshold=args.threshold)

    print("---predictions done---")


if __name__ == '__main__':
    main(sys.argv)