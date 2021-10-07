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
from dl4mic.unet import *

def main(argv):
    W = '\033[0m'  # white (normal)
    R = '\033[31m'  # red
    parser = ParserCreator.createArgumentParser("./evaluate.yml")
    args = parser.parse_args(argv[1:])
    full_QC_model_path = os.path.join(args.baseDir, args.name)
    if os.path.exists(os.path.join(full_QC_model_path, 'weights_best.hdf5')):
        print("The " + args.name + " network will be evaluated")
    else:
        print(R + '!! WARNING: The chosen model does not exist !!' + W)
        print('Please make sure you provide a valid model path and model name before proceeding further.')

    # Create a quality control/Prediction Folder
    prediction_QC_folder = os.path.join(full_QC_model_path, 'Quality Control', 'Prediction')
    if os.path.exists(prediction_QC_folder):
        shutil.rmtree(prediction_QC_folder)

    os.makedirs(prediction_QC_folder)

    # Load the model
    unet = load_model(os.path.join(full_QC_model_path, 'weights_best.hdf5'),
                      custom_objects={'_weighted_binary_crossentropy': weighted_binary_crossentropy(np.ones(2))})
    Input_size = unet.layers[0].output_shape[1:3]
    print('Model input size: '+str(Input_size[0])+'x'+str(Input_size[1]))

    # Create a list of sources
    source_dir_list = os.listdir(args.testInputPath)
    number_of_dataset = len(source_dir_list)
    print('Number of dataset found in the folder: '+str(number_of_dataset))

    predictions = []
    for i in range(number_of_dataset):
        print("processing dataset " + str(i+1) + ", file: " + source_dir_list[i])
        predictions.append(predict_as_tiles(os.path.join(args.testInputPath, source_dir_list[i]), unet))

    # Save the results in the folder along with the masks according to the set threshold
    saveResult(prediction_QC_folder, predictions, source_dir_list, prefix=prediction_prefix, threshold=None)

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

    print("---evaluation done---")


if __name__ == '__main__':
    main(sys.argv)
