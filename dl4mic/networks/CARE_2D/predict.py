# ------------------ Diclaimer ------------------
#WIP> DON'T TRY TO USE THIS NETWORK YET
# -----------------------------------------------
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
    parser = ParserCreator.createArgumentParser("./predict.yml")
    args = parser.parse_args(argv[1:])

    full_Prediction_model_path = os.path.join(args.baseDir, args.name)
    weight_extension = ""
    if isModelValid(full_Prediction_model_path,'.hdf5')
        weight_extension = '.hdf5'
    if isModelValid(full_Prediction_model_path,'.h5')
        weight_extension = '.h5'

    #?
    model = care(conf,name=args.name, basedir=args.baseDir)

    model.keras_model.summary();
    model.keras_model.load_weights(os.path.join(full_Prediction_model_path, 'weights_best'+weight_extension))

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


def isModelValid(full_model_path,weight_extension):
    (_,model_name) = os.path.split(full_model_path)
    if os.path.exists(os.path.join(full_model_path, 'weights_best'+weight_extension)):
        print("The " + model_name + " network will be used.")
        return True
    else:
        print('!! WARNING: The chosen model does not exist !!')
        print('Please make sure you provide a valid model path and model name before proceeding further.')
        return False

if __name__ == '__main__':
    main(sys.argv)