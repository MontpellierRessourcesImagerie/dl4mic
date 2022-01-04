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
from dl4mic.care import *
sys.stderr = sys.__stderr__

def main(argv):
    prediction_prefix = ""
    parser = ParserCreator.createArgumentParser("./predict.yml")
    args = parser.parse_args(argv[1:])

    model_weight_path = getValidModel(os.path.join(args.baseDir, args.name))

    model = createNetwork()
    
    model.keras_model.load_weights(model_weight_path)

    source_dir_list = prepareDataset()

    predictions = runPredictions(source_dir_list,model);
    
    saveResult(args.output, predictions, source_dir_list, prefix=prediction_prefix, threshold=args.threshold)

    print("---predictions done---")        
        
def getValidModel(full_model_path):
    (_,model_name) = os.path.split(full_model_path)
    if os.path.exists(os.path.join(full_model_path, 'weights_best.h5')):
        print("The " + model_name + " network will be used.")
        return os.path.join(full_model_path, 'weights_best.h5')
    else:
        if os.path.exists(os.path.join(full_model_path, 'weights_best.hdf5')):
            print("The " + model_name + " network will be used.")
            return os.path.join(full_model_path, 'weights_best.hdf5')
        else:
            print('!! WARNING: The chosen model does not exist !!')
            print('Please make sure you provide a valid model path and model name before proceeding further.')
            return False

def createNetwork():
    model = care(conf,name=args.name, basedir=args.baseDir)
    model.keras_model.summary()
    return model
    
def prepareDataset():
    source_dir_list = os.listdir(args.dataPath)
    number_of_dataset = len(source_dir_list)
    print('Number of dataset found in the folder: ' + str(number_of_dataset))
    return source_dir_list

def runPrediction(source_data_list,model):
    number_of_dataset = len(source_data_list)
    predictions = []
    for i in range(number_of_dataset):
        print("processing dataset " + str(i + 1) + ", file: " + source_data_list[i])
        predictions.append(predict_as_tiles(os.path.join(args.dataPath, source_data_list[i]), model))
    return predictions


if __name__ == '__main__':
    main(sys.argv)
