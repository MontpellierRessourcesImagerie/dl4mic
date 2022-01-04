import sys
import os
import time
import math
import warnings
import shutil
import csv
import tensorflow as tf
sys.path.append(r'../lib/')
from dl4mic.cliparser import ParserCreator
from dl4mic.unet import *

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

    data_gen_args = dict(width_shift_range=args.horizontalShift/100.,
                         height_shift_range=args.verticalShift/100.,
                         rotation_range=args.rotationRange,
                         zoom_range=args.zoomRange/100.,
                         shear_range=args.shearRange/100.,
                         horizontal_flip=args.horizontalFlip,
                         vertical_flip=args.verticalFlip,
                         validation_split=args.validationFraction/100,
                         fill_mode='reflect')
    print('Creating patches...')
    Patch_source, Patch_target = create_patches(args.dataSourcePath, args.dataTargetPath,
                                                args.patchSizeXY, args.patchSizeXY)

    (train_datagen, validation_datagen) = prepareGenerators(Patch_source, Patch_target, data_gen_args, args.batchSize,
                                                            target_size=(args.patchSizeXY, args.patchSizeXY))
    full_model_path = os.path.join(args.baseDir, args.name)
    W = '\033[0m'  # white (normal)
    R = '\033[31m'  # red
    if os.path.exists(full_model_path):
        print(R + '!! WARNING: Folder already exists and will be overwritten !!' + W)

    model_checkpoint = ModelCheckpoint(os.path.join(full_model_path, 'weights_best.hdf5'), monitor='val_loss',
                                       verbose=0, save_best_only=True)
    print('Getting class weights...')
    class_weights = getClassWeights(args.dataTargetPath)
    h5_file_path = None
    if(args.resumeTraining):
        h5_file_path = args.startingWeigth

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, verbose=1, mode='auto',
                                  patience=10, min_lr=0)
    model = unet(pretrained_weights=h5_file_path,
                 input_size=(args.patchSizeXY, args.patchSizeXY, 1),
                 pooling_steps=args.poolingSteps,
                 learning_rate=args.learningRate,
                 class_weights=class_weights)

    number_of_training_dataset = len(os.listdir(Patch_source))

    if args.stepsPerEpoch==0:
        number_of_steps = math.ceil((100 - args.validationFraction) / 100 * number_of_training_dataset / args.batchSize)
    else:
        number_of_steps = args.stepsPerEpoch

    validation_steps = max(1, math.ceil(args.validationFraction / 100 * number_of_training_dataset / args.batchSize))
    config_model = model.optimizer.get_config()
    print(config_model)
    if os.path.exists(full_model_path):
        print(R+'!! WARNING: Model folder already existed and has been removed !!'+W)
        shutil.rmtree(full_model_path)

    os.makedirs(full_model_path)
    os.makedirs(os.path.join(full_model_path, 'Quality Control'))

    # ------------------ Display ------------------
    print('---------------------------- Main training parameters ----------------------------')
    print('Number of epochs: '+str(args.epochs))
    print('Batch size: '+str(args.batchSize))
    print('Number of training dataset: '+str(number_of_training_dataset))
    print('Number of training steps: '+str(number_of_steps))
    print('Number of validation steps: '+str(validation_steps))
    print('---------------------------- ------------------------ ----------------------------')

    start = time.time()

    history = model.fit_generator(train_datagen,
                                  steps_per_epoch=number_of_steps,
                                  epochs=args.epochs,
                                  callbacks=[model_checkpoint, reduce_lr],
                                  validation_data=validation_datagen,
                                  validation_steps=validation_steps,
                                  shuffle=True, verbose=1)
    # Save the last model
    model.save(os.path.join(full_model_path, 'weights_last.hdf5'))

    lossDataCSVPath = os.path.join(full_model_path, 'Quality Control/training_evaluation.csv')
    with open(lossDataCSVPath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'val_loss', 'learning rate'])
        for i in range(len(history.history['loss'])):
            writer.writerow([history.history['loss'][i], history.history['val_loss'][i], history.history['lr'][i]])

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
