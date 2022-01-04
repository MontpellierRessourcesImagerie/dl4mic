# ------------------ Diclaimer ------------------
#WIP> DON'T TRY TO USE THIS NETWORK YET
# -----------------------------------------------
import sys
import os
import shutil
import csv
import io
sys.path.append(r'../lib/')
from dl4mic.cliparser import ParserCreator
from dl4mic.care import *

def main(argv):
    prediction_prefix = ""
    parser = ParserCreator.createArgumentParser("./evaluate.yml")
    args = parser.parse_args(argv[1:])
    
    full_QC_model_path = os.path.join(args.baseDir, args.name)
    model_weight_path = getValidModel(full_QC_model_path)

    # Create a quality control/Prediction Folder

    prediction_QC_folder = cleanAndGetFolderQC(full_QC_model_path)

    model = createNetwork()
    
    model.keras_model.load_weights(model_weight_path)

    source_dir_list = prepareDataset()

    predictions = runPredictions(source_dir_list,model)

    saveResult(prediction_QC_folder, predictions, source_dir_list, prefix=prediction_prefix, threshold=None)

    write_QC(full_QC_model_path)

    print("---evaluation done---")


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

def cleanAndGetFolderQC(full_QC_model_path):
    prediction_QC_folder = os.path.join(full_QC_model_path, 'Quality Control', 'Prediction')
    if os.path.exists(prediction_QC_folder):
        shutil.rmtree(prediction_QC_folder)
    os.makedirs(prediction_QC_folder)


def createNetwork():
    model = care(conf,name=args.name, basedir=args.baseDir)
    model.keras_model.summary()
    return model
        
def prepareDataset():
    source_dir_list = os.listdir(args.testInputPath)
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

    
def write_QC(full_QC_model_path):
    with open(os.path.join(full_QC_model_path, 'Quality Control', 'QC_metrics_' + args.name + '.csv'), "w",
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File name", "IoU", "IoU-optimised threshold"])

        # Initialise the lists
        filename_list = []
        best_threshold_list = []
        best_IoU_score_list = []

        for filename in os.listdir(args.testInputPath):

            if not os.path.isdir(os.path.join(args.testInputPath, filename)):
                print('Running QC on: ' + filename)
                test_input = io.imread(os.path.join(args.testInputPath, filename), as_gray=True)
                test_ground_truth_image = io.imread(os.path.join(args.testGroundTruthPath, filename), as_gray=True)

                (threshold_list, iou_scores_per_threshold) = getIoUvsThreshold(
                    os.path.join(prediction_QC_folder, prediction_prefix + filename),
                    os.path.join(args.testGroundTruthPath, filename))

                # Here we find which threshold yielded the highest IoU score for image n.
                best_IoU_score = max(iou_scores_per_threshold)
                best_threshold = iou_scores_per_threshold.index(best_IoU_score)

                # Write the results in the CSV file
                writer.writerow([filename, str(best_IoU_score), str(best_threshold)])

                # Here we append the best threshold and score to the lists
                filename_list.append(filename)
                best_IoU_score_list.append(best_IoU_score)
                best_threshold_list.append(best_threshold)


if __name__ == '__main__':
    main(sys.argv)
