var _CONDA_ENV = "dl";
var _INTERPRETER_PATH= newArray(
	getDir("home") + "anaconda3" + File.separator + "envs" + File.separator + _CONDA_ENV + File.separator + "bin" + File.separator +"python3",
	getDir("home") + "Anaconda3" + File.separator + "envs" + File.separator + _CONDA_ENV + File.separator + "python.exe",
	getDir("home") + ".conda" + File.separator + "envs" + File.separator + _CONDA_ENV + File.separator + "python.exe",
    getDir("home") + "mambaforge" + File.separator + "envs" + File.separator + _CONDA_ENV + File.separator + "bin" + File.separator + "python",
	'C:/ProgramData/Anaconda3/envs/dl/python.exe'
	)
var _PYTHON_INTERPRETER = findPythonInterpreter();
var _NETWORKS_DIR = getDirectory("imagej") + "dl4mic"+File.separator +"networks";
var _NETWORKS = initNetworksArray();
var _CURRENT_NETWORK = 'None';
var networkMenuItems = newMenu("Select a network Menu Tool", _NETWORKS);



var _LOADED_PARAMETERS = -1;
var _PT_TRAINING = 0;
var _PT_PREDICT = 1;
var _PT_EVALUATE = 2;

var _LOG_FILENAME = newArray(
		"log_training.txt",
		"log_prediction.txt",
		"log_evaluation.txt"
		);

var _PT_ACTION = newArray(
	"train",
	"predict",
	"evaluate"
	);

var _CURRENT_PARAMETER_GROUP 	= newArray(0);
var _CURRENT_PARAMETER_NAME		= newArray(0);
var _CURRENT_PARAMETER_DISPLAY 	= newArray(0);
var _CURRENT_PARAMETER_VALUE 	= newArray(0);
var _CURRENT_PARAMETER_HELP 	= newArray(0);
var _CURRENT_PARAMETER_TYPE 	= newArray(0);
var _CURRENT_PARAMETER_DEFAULT 	= newArray(0);

var trainingParameterMenuArray = newArray("user parameters","advanced parameters","internal network parameters","data augmentation","load parameters from file","---","python interpreter","install deep-learning env." );
var trainingParameterMenuItems = newMenu("Training Parameter Menu Tool", trainingParameterMenuArray);

var helpURL = "https://github.com/MontpellierRessourcesImagerie/dl4mic/wiki";

train();
exit;

//About Tool
macro "DL4Mic Action Tool (f2) - C666D00C555D10C444L2090C333Da0C777Db0CeeeDc0C777D01C666D11C555D21C444L31b1C888Dc1CeeeDd1C777L0222C555D32C666L42a2C555Lb2c2C999Dd2C777L0323C888D33CcccL43b3CaaaDc3C999Dd3C777L0424C999D34CdddL44b4CbbbDc4C999Dd4C777L0525C999D35CdddL45b5CbbbDc5C999Dd5C777L0626C999D36CdddL46b6CbbbDc6C999Dd6C777L0727C999D37CdddL47b7CbbbDc7C999Dd7C666D08C777L1828C999D38CdddL48b8CbbbDc8C999Dd8C666L0919C777D29C999D39CdddL49b9CbbbDc9C999Dd9CdddD0aC777D1aC666D2aC888D3aCdddL4abaCbbbDcaC999DdaCdddD1bC777D2bC888D3bCdddL4bbbCbbbDcbC999DdbCdddD2cC888D3cC999L4cbcC888DccC999DdcCeeeD3dCdddL4dcdCeeeDdd" {
	about();
}

macro "DL4Mic About [f2]" {
	about();
}

//Help Tool
macro "Help Action Tool (f3) - CcccD50C888D60C666L7080C888D90CcccDa0CdddD31C333D41C000L51a1C333Db1CdddDc1CbbbD22C000L3242C222D52C777D62CaaaL7282C777D92C222Da2C000Db2C111Dc2CbbbDd2CdddD13C000L2333C777D43CcccL7383C777Db3C000Dc3C111Dd3CdddDe3C333D14C000D24C777D34CdddD54C222D64C000L7484C333D94CeeeDa4C777Dc4C000Dd4C444De4CcccD05C000D15C222D25C444D55C000D65C333D75C222D85C000D95C666Da5C222Dd5C000De5CdddDf5C888D06C000D16C777D26C555D56C333D66CeeeD86C000D96C222Da6C777Dd6C000De6C888Df6C666D07C000D17CaaaD27CdddD77C444D87C000D97C444Da7CaaaDd7C000De7C666Df7D08C000D18CaaaD28C111D78C000D88C222D98CeeeDa8CaaaDd8C000De8C666Df8C888D09C000D19C777D29C888L7989C777Dd9C000De9C888Df9CcccD0aC000D1aC222D2aC999L7a8aC222DdaC000DeaCdddDfaC333D1bC000D2bC777D3bC111L7b8bC777DcbC000DdbC444DebCdddD1cC111D2cC000D3cC777D4cCdddL7c8cC777DbcC000DccC111DdcCdddDecCbbbD2dC111D3dC000D4dC222D5dC777D6dCaaaL7d8dC777D9dC222DadC000DbdC111DcdCbbbDddCdddD3eC333D4eC000L5eaeC444DbeCdddDceD5fC888D6fC666L7f8fC888D9fCdddDaf" {
	showHelp();	
}

macro "DL4mic Help [f3]" {
	showHelp();
}

//Select Network Tool
macro "Select a network Menu Tool - C888L7080C777D51C111D61C000L7181C111D91C777Da1D32C111D42C000L5262C555L7282C000L92a2C111Db2C777Dc2D13C111D23C000L3343C555D53CdddD63D93C555Da3C000Lb3c3C111Dd3C777De3D04C000L1424C222D34CdddD44Db4C222Dc4C000Ld4e4C777Df4CcccD05C222D15C000L2535C222D45CaaaD55Da5C222Db5C000Lc5d5C222De5CcccDf5CaaaD26C222D36C000L4656C222D66CaaaL7686C222D96C000La6b6C222Dc6CaaaDd6C999D07C111D17C777D27CaaaD47C222D57C000L6797C222Da7CaaaDb7C777Dd7C111De7C999Df7D08C000L1828C111D38C777D48CaaaD68C333L7888CaaaD98C777Db8C111Dc8C000Ld8e8C999Df8CdddD19C555D29C000L3949C111D59C777D69CeeeD79C888D99C111Da9C000Lb9c9C444Dd9CdddDe9CcccD0aC666D1aCdddL2a3aC555D4aC000L5a6aC111L7a8aC000L9aaaC555DbaCdddLcadaC666DeaCcccDfaC777D0bC000L1b2bC555D3bCdddL4b5bC555D6bC000L7b8bC555D9bCdddLabbbC555DcbC000LdbebC777DfbD1cC111D2cC000L3c4cC555D5cCdddL6c9cC555DacC000LbcccC111DdcC777DecD3dC111D4dC000L5d6dC555L7d8dC000L9dadC111DbdC777DcdD5eC111D6eC000L7e8eC111D9eC777DaeC888L7f8f" {
	selectNetwork();	
}

//Network Info Tool
macro "Info Action Tool (f4) - CcccD50C888D60C666L7080C888D90CcccDa0CdddD31C333D41C000L51a1C333Db1CdddDc1CbbbD22C000L3242C222D52C777D62CaaaL7282C777D92C222Da2C000Db2C111Dc2CbbbDd2CdddD13C000L2333C777D43Db3C000Dc3C111Dd3CdddDe3C333D14C000D24C777D34C999L7484C777Dc4C000Dd4C444De4CcccD05C000D15C222D25C111L7585C222Dd5C000De5CdddDf5C888D06C000D16C777D26CdddL7686C777Dd6C000De6C888Df6C666D07C000D17CaaaD27C444L7787CaaaDd7C000De7C666Df7D08C000D18CaaaD28C000L7888CaaaDd8C000De8C666Df8C888D09C000D19C777D29C000L7989C777Dd9C000De9C888Df9CcccD0aC000D1aC222D2aC000L7a8aC222DdaC000DeaCdddDfaC333D1bC000D2bC777D3bC999L7b8bC777DcbC000DdbC444DebCdddD1cC111D2cC000D3cC777D4cDbcC000DccC111DdcCdddDecCbbbD2dC111D3dC000D4dC222D5dC777D6dCaaaL7d8dC777D9dC222DadC000DbdC111DcdCbbbDddCdddD3eC333D4eC000L5eaeC444DbeCdddDceD5fC888D6fC666L7f8fC888D9fCdddDaf" {
	info();
}

macro "DL4Mic Info [f4]" {
	info();
}

//Training Parameters Tool
macro "Training Parameter Menu Tool - C555D60C000L7080C555D90CeeeD21C888D31CaaaD41CcccD51C000D61C111L7181C000D91CcccDa1C999Db1C888Dc1CdddDd1D12C111D22C000L3242C111D52C000D62C777L7282C000D92C111Da2C000Lb2c2C111Dd2CeeeDe2C888D13C000D23C444D33C333D43C000D53C222D63CeeeL7383C333D93C000Da3C444Lb3c3C000Dd3C999De3D14C000D24C444D34CeeeD54C333Dc4C000Dd4CaaaDe4CdddD15C111D25C000D35C777D65C222L7585C777D95CeeeDb5C000Dc5C111Dd5CcccDe5C666D06C000L1626C333D36C777D56C000L6696C777Da6C222Dc6C000Ld6e6C555Df6C000D07C111D17C777D27CeeeD37C222D57C000D67CcccL7787C000D97C222Da7CeeeDc7C777Dd7C111De7C000Df7D08C222D18C888D28CeeeD38C222D58C000D68CcccL7888C000D98C222Da8CeeeDc8C777Dd8C111De8C000Df8C444D09C000L1929C333D39C777D59C000L6999C888Da9C333Dc9C000Ld9e9C555Df9CbbbD1aC111D2aC000D3aCeeeD4aC777D6aC222L7a8aC888D9aC000DcaC111DdaCcccDeaCaaaD1bC000D2bC333D3bCeeeDabC444DcbC000DdbC999DebD1cC000D2cC444L3c4cC000D5cC333D6cCeeeL7c8cC222D9cC000DacC333DbcC444DccC000DdcC888DecCeeeD1dC111D2dC000L3d4dC111D5dC000D6dC777D7dC888D8dC000D9dC111DadC000LbdcdC111DddCdddDedD2eC888D3eC999D4eCdddD5eC000D6eC111D7eC222D8eC000D9eCbbbDaeCaaaDbeC888DceCeeeDdeC666D6fC000L7f8fC444D9f" {	
	showMenuParameterDialog();
}

//Train Tool
macro "Train the model Action Tool (f6) - C999D01C000L1141C111D51C444D61CcccL7181C444D91C111Da1C000Lb1e1C999Df1C555D02C000L12e2C555Df2D03C000D13CbbbD23C777D63C000L7383C777D93CbbbDd3C000De3C555Df3D04C000D14CbbbD24C000L7484CbbbDd4C000De4C555Df4D05C000D15CbbbD25C000L7585CbbbDd5C000De5C555Df5D06C000D16CbbbD26C000L7686CbbbDd6C000De6C555Df6D07C000D17CbbbD27C000L7787CbbbDd7C000De7C555Df7D08C000D18CbbbD28C000L7888CbbbDd8C000De8C555Df8D09C000D19CbbbD29C000L7989CbbbDd9C000De9C555Df9D0aC000D1aCbbbD2aC000L7a8aCbbbDdaC000DeaC555DfaD0bC000L1b5bC222D6bC000L7b8bC222D9bC000LabebC555DfbC999D0cC000L1cecC999DfcCcccD6dC000L7d8dCcccD9dC111L7e8e" {
	train();
}

macro "DL4Mic Train [f6]" {
	train();
}

//Evaluate Tool
macro "Evaluate the model Action Tool (f7) - CcccD50C888D60C666L7080C888D90CcccDa0CdddD31C333D41C000L51a1C777Db1CcccDe1CbbbD22C000L3242C222D52C777D62CaaaL7282C777D92C333Da2C999Db2C777Dd2C000De2C777Df2CdddD13C000L2333C777D43Dc3C000Dd3C111De3CcccDf3C333D14C000D24C777D34Db4C000Lc4d4CbbbDe4CcccD05C000D15C222D25C777Da5C000Lb5c5CbbbDd5C888D06C000D16C777D26C999L5666C777D96C000Da6C111Db6CbbbDc6C777De6CdddDf6C666D07C000D17CaaaD27C111D57C000D67C777L7787C000D97C111Da7CbbbDb7Dd7C000De7C666Df7D08C000D18CaaaD28CbbbD58C111D68C000L7888C111D98CbbbDa8CaaaDd8C000De8C666Df8C888D09C000D19C777D29CbbbD69C111L7989CbbbD99C777Dd9C000De9C888Df9CcccD0aC000D1aC222D2aCdddL7a8aC111DdaC000DeaCcccDfaC333D1bC000D2bC777D3bDcbC000DdbC333DebCdddD1cC000L2c3cC777D4cDbcC000DccC111DdcCdddDecCbbbD2dC000L3d4dC111D5dC777D6dCaaaL7d8dC777D9dC222DadC000DbdC111DcdCbbbDddCdddD3eC333D4eC000L5eaeC333DbeCdddDceCcccD5fC888D6fC666L7f8fC888D9fCcccDaf" {
	evaluate();
}

macro "DL4Mic Evaluate [f7]" {
	evaluate();
}

macro "Evaluate the model Action Tool (f7) Options" {
	showParametersDialog(_PT_EVALUATE,"user_parameters");
}

//Predict Tool
macro "Predict Action Tool (f8) - C999D21C111D31C999D41C555D22C000L3242C333D52CdddD62C555D23C000L3363C888D73C555D24C000D34CaaaD44C777D54C000L6474C222D84CcccD94C555D25C000D35CbbbD45CcccD65C333D75C000L8595C666Da5C555D26C000D36CbbbD46C999D86C000L96a6C222Db6CbbbDc6C555D27C000D37CbbbD47CdddD97C333Da7C000Lb7c7C999Dd7C555D28C000D38CbbbD48CdddD98C333Da8C000Lb8c8C999Dd8C555D29C000D39CbbbD49C999D89C000L99a9C222Db9CbbbDc9C555D2aC000D3aCbbbD4aCcccD6aC333D7aC000L8a9aC666DaaC555D2bC000D3bCaaaD4bC777D5bC000L6b7bC222D8bCcccD9bC555D2cC000L3c6cC888D7cC555D2dC000L3d4dC333D5dCdddD6dC999D2eC111D3eC999D4e" {
	predict();
}

macro "DL4Mic Predict [f8]" {
	predict();
}

macro "Predict Action Tool (f8) Options" {
	showParametersDialog(_PT_PREDICT,"user_parameters");
}

//Supplementary Macros, not associated with tools
macro "Display Training Evaluation Plot []"{
	baseDir = getDirectory("Base Directory");
	files = getFileList(baseDir);
	
	Dialog.create("Select the model");
	Dialog.addChoice("Model",files,"");
	Dialog.show();
	model = Dialog.getChoice();
	
	displayTrainingEvaluationPlot(baseDir,model);
}

macro "Open Log []"{
	_pt=askLogToOpen();
	catchLog(_pt);
}

macro "Restore all parameters to default values []"{
	showMessageWithCancel("Are you sure you want to Restore all parameters to default? This cannot be undone");
	current_network = _CURRENT_NETWORK;
	for(n=0;n<_NETWORKS.length();n++){
		_CURRENT_NETWORK=_NETWORKS[n];
		for(_pt = 0 ; _pt < 3;_pt++){
			readParameters(_pt);
			for (i = 0; i < _CURRENT_PARAMETER_GROUP.length; i++) {
				_CURRENT_PARAMETER_VALUE[i] = _CURRENT_PARAMETER_DEFAULT[i];
			}
			saveParameters(_pt);
		}
	}
	_CURRENT_NETWORK = current_network;
}

//Display information

function showHelp() {
	run('URL...', 'url='+helpURL);
}

function about() {
	aboutMessage = "<html><h1>About DL4Mic</h1>" +
	  "<p>	(c) 2020, INSERM <br>" +
      "written by Volker Baecker at Montpellier Ressources Imagerie, Biocampus Montpellier, INSERM, CNRS, University of Montpellier (www.mri.cnrs.fr)" +
      "<p>DL4Mic is free software under the <a href='https://raw.githubusercontent.com/MontpellierRessourcesImagerie/dl4mic/master/LICENSE.txt'>CeCILL-B license</a>." +
	  "<p>The python code of the neural networks has been adapted from the <a href='https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki'>ZeroCostDL4Mic</a> project." +
	  "<p>Please also cite this original paper of a given network when using it via this software." +
	  "<p>Have fun !!!" +
      "</html>";

	showMessage("About DL4Mic Toolset", aboutMessage);
}

function info() {
	requireNetwork();
	message = File.openAsString(_NETWORKS_DIR + File.separator + _CURRENT_NETWORK + File.separator +"info.html");
	imagePath = "file:///"+_NETWORKS_DIR + File.separator + _CURRENT_NETWORK + File.separator +"picture.png";
	imagePath = replace(imagePath, '\\', File.separator);
	message = replace(message, "<!--img-->", "<img src='"+imagePath+"'>");
	showMessage("Info: "+_CURRENT_NETWORK, message);
}


//Main Functions

function train() {
	requireNetwork();

	epochs = getValueOfParameter(_PT_TRAINING,'epochs');
	dataPath = getValueOfParameter(_PT_TRAINING,'dataPath');
	if (dataPath == '') {
		dataPath = getValueOfParameter(_PT_TRAINING,'dataSourcePath');
	}
	baseDir = getValueOfParameter(_PT_TRAINING,'baseDir');
	name = getValueOfParameter(_PT_TRAINING,'name');
	outPath = baseDir + name;
	//showMessageWithCancel("Training of "+_CURRENT_NETWORK+"\n"+"Epochs: " + epochs + "\n" + "Data path: "+dataPath + "\n" + "Result will be saved as " + outPath);
	
	launchPythonExecution(_PT_TRAINING);
	catchLog(_PT_TRAINING);
	
	displayTrainingEvaluationPlot(baseDir,name);
	saveParameterString(_PT_TRAINING);
	
}

function evaluate() {
	inputFolder = "";
	outputFolder = "";
	workingOnOpenImage = false;
	isStack = false;
	if (nImages==0) {
		inputFolder = getDirectory("Input Folder");	
		outputFolder = getDirectory("Ground Truth Folder");	
	} else {
		workingOnOpenImage = true;
		emptyTmpFolder();
		title = getTitle();
		parts = split(title, ".");
		title = parts[0] + ".tif";
		File.makeDirectory(getDirectory("imagej") + "dl4mic" + File.separator + "tmp");
		inputFolder = getDirectory("imagej") + "dl4mic" + File.separator + "tmp" + File.separator + "in";
		File.makeDirectory(inputFolder);
		outputFolder = getDirectory("imagej") + "dl4mic" + File.separator + "tmp" + File.separator + "out";
		if (nSlices>1) {
			isStack = true;
			run("Image Sequence... ", "format=TIFF save="+inputFolder+File.separator+title);	
		} else {
			saveAs("tiff", inputFolder +File.separator+title);
		}
	}
	readParameters(_PT_EVALUATE);
	for (i = 0; i < _CURRENT_PARAMETER_GROUP.length; i++) {
		if (_CURRENT_PARAMETER_NAME[i]=='testInputPath') _CURRENT_PARAMETER_VALUE[i] = inputFolder;
		if (_CURRENT_PARAMETER_NAME[i]=='testGroundTruthPath') _CURRENT_PARAMETER_VALUE[i] = outputFolder;
	}
	saveParameters(_PT_EVALUATE);
	
	launchPythonExecution(_PT_EVALUATE);
	catchLog(_PT_EVALUATE);
	
	files = getFileList(outputFolder);
	count = 0;
	for (i = 0; i < files.length; i++) {
		if (endsWith(files[i], ".tif")) {
			count++;
		}
	}

	baseDir = getValueOfParameter(_PT_EVALUATE,'baseDir');
	name = getValueOfParameter(_PT_EVALUATE,"name");
	qcDir = baseDir + name + File.separator +"Quality Control"+File.separator ;
	predictionsDir = qcDir + "Prediction"+File.separator ;
	
	print("" + count + " result images have been written to: \n" + predictionsDir);

	targetID = openImageSeries(outputFolder, "Target", "", 2);
	sourceID = openImageSeries(inputFolder, "Source", "", 2);
	predictionID = openImageSeries(predictionsDir, "Prediction", "", 2);
	
	
	Table.open(qcDir+File.separator+'QC_metrics_'+name+'.csv') ;
	
	wait(500);
	parts = split(Table.headings, "\t");
	if (parts[0] == "image #") {
		Table.renameColumn('image #', 'image');
		Table.sort('image');
	}
	for(i=1 ; i < parts.length ; i++){
		if (parts[i] == "IoU-optimised threshold") {
			//Threshold the predicted image and plot IoU vs. Threshold 
			selectImage("Prediction");
			images = Table.getColumn("File name");
			thresholds = Table.getColumn("IoU-optimised threshold");
			positions = Array.rankPositions(images);
			for (i = 0; i < positions.length; i++) {
				Stack.setSlice(i+1);
				if(thresholds[positions[i]]>255){
					threshold = 255-(thresholds[positions[i]]-255);
					run("Macro...", "code=v=(v<="+threshold+")*255 slice");	
				}else{
					run("Macro...", "code=v=(v>"+thresholds[positions[i]]+")*255 slice");	
				}
			}
			run("Invert LUT");
			Stack.setSlice(1);
			ious = Table.getColumn("IoU");
			Plot.create("IoU vs. Threshold", "IoU", "IoU-optimised threshold");
			Plot.add("Circle", Table.getColumn("IoU", 'QC_metrics_'+name+'.csv'), Table.getColumn("IoU-optimised threshold", 'QC_metrics_'+name+'.csv'));
			Plot.setStyle(0, "blue,#a0a0ff,1.0,Circle");

			Array.getStatistics(thresholds, min, max, mean, stdDev);
			
			print("-----------------------");
			print("Average best threshold:", mean);
			print("-----------------------");
		}
	}
	if (containsFileWithPrefix(qcDir+File.separator, "SSIM_GTvsSource_")) {
		ssimTargetVSSourceID = openImageSeries(qcDir+File.separator, "Target vs. Source SSIM", "SSIM_GTvsSource_", 2);
		setSubtitles(ssimTargetVSSourceID, "Input v. GT mSSIM", "Target vs. Source SSIM:");
		run("Fire");
		run("Calibration Bar...", "location=[Lower Right] fill=White label=Black number=6 decimal=1 font=12 zoom=1.4 overlay");
	}
	if (containsFileWithPrefix(qcDir+File.separator, "SSIM_GTvsPrediction_")) {
		ssimTargetVSPredictionID = openImageSeries(qcDir+File.separator, "Target vs. Prediction SSIM", "SSIM_GTvsPrediction_", 2);
		setSubtitles(ssimTargetVSPredictionID, "Prediction v. GT mSSIM", "Target vs. Prediction SSIM:");
		run("Fire");
		run("Calibration Bar...", "location=[Lower Right] fill=White label=Black number=6 decimal=1 font=12 zoom=1.4 overlay");
	}
	if (containsFileWithPrefix(qcDir+File.separator, "RSE_GTvsSource_")) {
		rseTragetVSSourceID = openImageSeries(qcDir+File.separator, "Target vs. Source NRMSE", "RSE_GTvsSource_", 2);
		setSubtitles(rseTragetVSSourceID, "Input v. GT NRMSE", "Target vs. Source NRMSE:");
		run("Fire");
		run("Calibration Bar...", "location=[Lower Right] fill=White label=Black number=6 decimal=1 font=12 zoom=1.4 overlay");
	}
	if (containsFileWithPrefix(qcDir+File.separator, "RSE_GTvsPrediction_")) {
		rseTragetVSPredictionID = openImageSeries(qcDir+File.separator, "Target vs. Prediction NRMSE", "RSE_GTvsPrediction_", 2);
		setSubtitles(rseTragetVSPredictionID, "Prediction v. GT NRMSE", "Target vs. Prediction NRMSE:");
		run("Fire");
		run("Calibration Bar...", "location=[Lower Right] fill=White label=Black number=6 decimal=1 font=12 zoom=1.4 overlay");
	}
	run("Tile");
} 

function predict() {
	inputFolder = "";
	outputFolder = "";
	workingOnOpenImage = false;
	isStack = false;
	nrOfImages = 0;
	if (nImages==0) {
		inputFolder = getDirectory("Input Folder");	
		outputFolder = getDirectory("Output Folder");	
		files = getFileList(inputFolder);
		nrOfImages = files.length;
	} else {
		workingOnOpenImage = true;
		emptyTmpFolder();
		title = getTitle();
		parts = split(title, ".");
		title = parts[0] + ".tif";
		File.makeDirectory(getDirectory("imagej") + "dl4mic" + File.separator + "tmp");
		inputFolder = getDirectory("imagej") + "dl4mic" + File.separator + "tmp" + File.separator + "in";
		File.makeDirectory(inputFolder);
		outputFolder = getDirectory("imagej") + "dl4mic" + File.separator + "tmp" + File.separator + "out";
		File.makeDirectory(outputFolder);
		nrOfImages = nSlices;
		if (nSlices>1) {
			isStack = true;
			run("Image Sequence... ", "format=TIFF save="+inputFolder);	
		} else {
			saveAs("tiff", inputFolder +File.separator+title);
		}
	}
	readParameters(_PT_PREDICT);
	for (i = 0; i < _CURRENT_PARAMETER_GROUP.length; i++) {
		if (_CURRENT_PARAMETER_NAME[i]=='dataPath') _CURRENT_PARAMETER_VALUE[i] = inputFolder;
		if (_CURRENT_PARAMETER_NAME[i]=='output') _CURRENT_PARAMETER_VALUE[i] = outputFolder;
	}
	saveParameters(_PT_PREDICT);

	
	launchPythonExecution(_PT_PREDICT);
	catchLog(_PT_PREDICT);
	
	files = getFileList(outputFolder);
	count = 0;
	for (i = 0; i < files.length; i++) {
		if (endsWith(files[i], ".tif")) {
			count++;
		}
	}

	print("" + count + " result images have been written to: \n" + outputFolder);
	if (workingOnOpenImage) {
		if (isStack) {
			run("Image Sequence...", "select=" + outputFolder + " dir=" + outputFolder + " sort");
			resID = getImageID();
			resSlices = nSlices;
			if (resSlices>nrOfImages) {
				run("Duplicate...", "duplicate range=1-"+resSlices/2);
				selectImage(resID);
				run("Duplicate...", "duplicate range="+((resSlices/2)+1)+"-"+resSlices);
				selectImage(resID);
				close();
			}
		} else {
			open(outputFolder +File.separator+title);
		}
	}
}


//Python Management

function installEnv(oldEnv, newEnv) {
	os = toLowerCase(getInfo("os.name"));
	if (indexOf(os, "win")>-1) {
		print("Installing a window env");
		a = exec("cmd", "/c", "start", "cmd", "/c", "conda env remove -y -n "+newEnv+" & sleep 5");
		envFile = getDirectory("imagej") + "dl4mic"+File.separator+"environment_win.yml";
		
		print("Env File ="+ envFile);
		envContent = File.openAsString(envFile);
		envContent = replace(envContent, "name: "+oldEnv, "name: "+newEnv);
		File.saveString(envContent, envFile);
		envPosition = "C:"+File.separator+"ProgramData"+File.separator+"Anaconda3"+File.separator+"envs"+File.separator+newEnv;
		
		print("Env Position ="+ envPosition);
		exec("cmd", "/c", "start", "cmd", "/c", "conda env create -p "+envPosition+" -f "+envFile+" & sleep 3");
		toolsetFile = getDirectory("imagej") + "macros"+File.separator+"toolsets"+File.separator+"DL4Mic_Toolset.ijm";
		toolsetContent = File.openAsString(toolsetFile);
		toolsetContent = replace(toolsetContent, 'var _CONDA_ENV = "'+oldEnv+'";', 'var _CONDA_ENV = "'+newEnv+'";');
		File.saveString(toolsetContent, toolsetFile);
		exec("cmd", "/c", "start", "cmd", "/c", "conda init cmd.exe");
	} else {
		cmd = getDirectory("home")+"anaconda3/condabin/conda";
		command = cmd+" env remove -y -n "+newEnv;
		print(command);
		a = exec("gnome-terminal", "--geometry=80x20", "--", "bash", "-c", command);	
		print(a);
		envFile = getDirectory("imagej") + "dl4mic/environment.yml";
		envContent = File.openAsString(envFile);
		envContent = replace(envContent, "name: "+oldEnv, "name: "+newEnv);
		File.saveString(envContent, envFile);
		command = cmd+" env create -f " + envFile;
		print(command);
		a = exec("gnome-terminal", "--geometry=80x20", "--", "bash", "-c", command);	
		print(a);
		toolsetFile = getDirectory("imagej") + "macros/toolsets/DL4Mic_Toolset.ijm";
		toolsetContent = File.openAsString(toolsetFile);
		toolsetContent = replace(toolsetContent, 'var _CONDA_ENV = "'+oldEnv+'";', 'var _CONDA_ENV = "'+newEnv+'";');
		File.saveString(toolsetContent, toolsetFile);
	}
}

function launchPythonExecution(_PT){
	requireNetwork();
	requireInboundPT(_PT);
	
	if (!File.exists(_PYTHON_INTERPRETER)) {
		showMessage("Error", "Could not find the python environment. Please install the dl4mic-python-environment or set the path to the interpreter.");
		return;
	}
	
	baseFolder = _NETWORKS_DIR + File.separator + _CURRENT_NETWORK;
	script = _PT_ACTION[_PT]+".py";
	parameters = getParameterString(_PT);
	logPath = _NETWORKS_DIR + File.separator + _CURRENT_NETWORK + File.separator + _LOG_FILENAME[_PT];
	File.delete(logPath);
	os = toLowerCase(getInfo("os.name"));
	if (indexOf(os, "win")>-1) {
		writeBatchFile(_PT,parameters);
		setOption("WaitForCompletion", false);
		folder = getDirectory("imagej");
		a = exec(folder + File.separator + _PT_ACTION[_PT] + ".bat");
	} else {
		command = "cd "+baseFolder+"; "+_PYTHON_INTERPRETER+" -u "+script+" "+parameters+" 2>&1 | tee "+ _LOG_FILENAME[_PT];
		print(command);
		a = exec("gnome-terminal", "--geometry=0x0", "-x", "sh", "-c", command);
	}
}

function writeBatchFile(_PT,parameters) {
	requireNetwork();
	requireInboundPT(_PT);
	
	parameters = getParameterString(_PT);
	dir = getDirectory("imagej");
	parts = split(dir, "\\");
	driveLetter = parts[0];
	baseFolder = _NETWORKS_DIR + File.separator + _CURRENT_NETWORK;
	
	command = "conda activate "+_CONDA_ENV+" && "+driveLetter+" && cd "+baseFolder+" && python.exe -u "+_PT_ACTION[_PT]+".py "+parameters+" > "+ _LOG_FILENAME[_PT]+" 2>&1";
	File.saveString(command, dir+_PT_ACTION[_PT]+".bat");
}


//Parameters Management 

function getValueOfParameter(_PT,aParameter) {
	requireInboundPT(_PT);

	if(_LOADED_PARAMETERS != _PT){
		readParameters(_PT);
	}
	result = '';
	for (i = 0; i < _CURRENT_PARAMETER_NAME.length; i++) {
		if (_CURRENT_PARAMETER_NAME[i]==aParameter) {
			return _CURRENT_PARAMETER_VALUE[i];
		}
	}
	return result;
}

macro "Save Parameter String"{
	Dialog.create("Chose Parameter Typ");
	items = newArray(_PT_TRAINING,_PT_EVALUATE,_PT_PREDICT);
	Dialog.addChoice("Type", items, _PT_TRAINING);
	Dialog.show();
	choice = Dialog.getChoice();
	string = getParameterString(choice);
	print(string);
	saveParameterString(choice);
}


function getParameterString(_PT){
	requireInboundPT(_PT);
	
	if(_LOADED_PARAMETERS != _PT){
		readParameters(_PT);
	}
	string = "";
	for (i = 0; i < _CURRENT_PARAMETER_GROUP.length; i++) {
		if (_CURRENT_PARAMETER_TYPE[i]=="bool") {
			if (_CURRENT_PARAMETER_VALUE[i]=='True' || _CURRENT_PARAMETER_VALUE[i]=='1') {
				string = string + "--" + _CURRENT_PARAMETER_NAME[i] + " ";
			}
		} else {
			string = string + "--" + _CURRENT_PARAMETER_NAME[i] + " " + _CURRENT_PARAMETER_VALUE[i] + " ";
		}
	}
	return string;
}


function loadParameterFromFile(filePath){
	if(!File.exists(filePath)){
		print("Parameter file doesn't exists, Loading failed");
		return;
	}
	fileContent = File.openAsString(filePath);
	fields = split(fileContent, "(--)");
	networkField = split(fields[1], " ");
	setNetwork(networkField[1]);
	readParameters(_PT_TRAINING);
	
	for (f = 2; f < fields.length; f++) {
		currentField = split(fields[f], " ");
		for(p = 0 ; p < _CURRENT_PARAMETER_NAME.length;p++){
			if(currentField[0] != _CURRENT_PARAMETER_NAME[p]){
				continue;
			}
			if(_CURRENT_PARAMETER_TYPE[p] == "bool"){
				 _CURRENT_PARAMETER_VALUE[p] = true;
			}
			else{
				 _CURRENT_PARAMETER_VALUE[p] = currentField[1];
			}
			break;
		}
	}
	saveParameters(_PT_TRAINING);
}

function readParameters(_PT){
	requireNetwork();
	requireInboundPT(_PT);

	baseFolder = _NETWORKS_DIR + File.separator + _CURRENT_NETWORK;
	_CURRENT_PARAMETER_GROUP = newArray(0);
	_CURRENT_PARAMETER_NAME = newArray(0);
	_CURRENT_PARAMETER_DISPLAY = newArray(0);
	_CURRENT_PARAMETER_VALUE = newArray(0);
	_CURRENT_PARAMETER_HELP = newArray(0);
	_CURRENT_PARAMETER_DEFAULT = newArray(0);
	_CURRENT_PARAMETER_TYPE = newArray(0);

	//PT Dependent
	parameterFile = File.openAsString(baseFolder + File.separator+_PT_ACTION[_PT]+".yml");

	parameterLines = split(parameterFile, "\n");
	for (i = 0; i < parameterLines.length; i++) {
		line = String.trim(parameterLines[i]);
		if (line.length<1) continue;
		nameAndValue = split(line, ":");
		if (nameAndValue.length==1) {
			currentGroup = replace(line, ":", '');    
			continue;
		} 
		name = replace(nameAndValue[0], "-", "");
		name = String.trim(name);

		value = String.trim(nameAndValue[1]);
		currentValue = 2;
		while(currentValue<nameAndValue.length){
			value += ":"+String.trim(nameAndValue[currentValue]);
			currentValue++;
		}
		if (name=='name') {
			equalizesParametersLength();
			_CURRENT_PARAMETER_NAME  = Array.concat(_CURRENT_PARAMETER_NAME  , value);
			_CURRENT_PARAMETER_GROUP = Array.concat(_CURRENT_PARAMETER_GROUP, currentGroup);
		}
		if (name=='display') _CURRENT_PARAMETER_DISPLAY = Array.concat(_CURRENT_PARAMETER_DISPLAY, value);
		if (name=='value')   _CURRENT_PARAMETER_VALUE   = Array.concat(_CURRENT_PARAMETER_VALUE  , value);
		if (name=='type')    _CURRENT_PARAMETER_TYPE  = Array.concat(_CURRENT_PARAMETER_TYPE   , value);
		if (name=='default') _CURRENT_PARAMETER_DEFAULT = Array.concat(_CURRENT_PARAMETER_DEFAULT, value);
		if (name=='help')  _CURRENT_PARAMETER_HELP  = Array.concat(_CURRENT_PARAMETER_HELP   , value);
	}
	equalizesParametersLength();
	_LOADED_PARAMETERS = _PT;
}

function saveParameters(_PT){
	requireNetwork();
	requireInboundPT(_PT);
	
	baseFolder = _NETWORKS_DIR + File.separator + _CURRENT_NETWORK;
	group = "";
	content = "";
	for (i = 0; i < _CURRENT_PARAMETER_GROUP.length; i++) {
		if (group!= _CURRENT_PARAMETER_GROUP[i]) {
			group = _CURRENT_PARAMETER_GROUP[i];
			content = content + group + ":\n";
		}
		content = content + '- name: ' 		+ _CURRENT_PARAMETER_NAME[i] + "\n";
		content = content + '  display: ' 	+ _CURRENT_PARAMETER_DISPLAY[i] + "\n";
		if(_CURRENT_PARAMETER_VALUE[i] == ""){
			content = content + '  value: ' 	+ _CURRENT_PARAMETER_DEFAULT[i] + "\n";
		}else{
			content = content + '  value: ' 	+ _CURRENT_PARAMETER_VALUE[i] + "\n";
		}
		content = content + '  type: ' 		+ _CURRENT_PARAMETER_TYPE[i] + "\n";
		content = content + '  default: '	+ _CURRENT_PARAMETER_DEFAULT[i] + "\n";
		content = content + '  help: ' 	  	+ _CURRENT_PARAMETER_HELP[i] + "\n";
	}
	
	File.saveString(content, baseFolder + File.separator+_PT_ACTION[_PT]+".yml");
}

function equalizesParametersLength(){
	//Always specify at least value or default
	if( _CURRENT_PARAMETER_DISPLAY.length < _CURRENT_PARAMETER_NAME.length ){
		nullDisplay = _CURRENT_PARAMETER_NAME[_CURRENT_PARAMETER_NAME.length-1];
		_CURRENT_PARAMETER_DISPLAY = Array.concat(_CURRENT_PARAMETER_DISPLAY, nullDisplay);
	}
	if( _CURRENT_PARAMETER_VALUE.length < _CURRENT_PARAMETER_NAME.length ){ 
		nullValue = _CURRENT_PARAMETER_DEFAULT[_CURRENT_PARAMETER_DEFAULT.length-1];//If no values were specified, use the Default one
		_CURRENT_PARAMETER_VALUE = Array.concat(_CURRENT_PARAMETER_VALUE, nullValue);
	}
	if( _CURRENT_PARAMETER_TYPE.length < _CURRENT_PARAMETER_NAME.length ){
		nullType = "NullType";
		_CURRENT_PARAMETER_TYPE = Array.concat(_CURRENT_PARAMETER_TYPE, nullType);
	}
	if( _CURRENT_PARAMETER_DEFAULT.length < _CURRENT_PARAMETER_NAME.length ){
		nullDefault = _CURRENT_PARAMETER_VALUE[_CURRENT_PARAMETER_VALUE.length-1];  //if no default values were specified, use the "value" one
		_CURRENT_PARAMETER_DEFAULT = Array.concat(_CURRENT_PARAMETER_DEFAULT, nullDefault);
	}
	if( _CURRENT_PARAMETER_HELP.length < _CURRENT_PARAMETER_NAME.length ){
		nullHelp = "NullHelp";
		_CURRENT_PARAMETER_HELP = Array.concat(_CURRENT_PARAMETER_HELP, nullHelp);
	}
	}


//Dialog Management

function showMenuParameterDialog() {
	item = getArgument();
	if (item == 'python interpreter') {
		showPythonInterpreterDialog();
		return;
	}
	if (item == 'install deep-learning env.') {
		showInstallEnvDialog();
		return;	
	}
	if (item == 'load parameters from file') {
		showLoadParametersDialog();
		return;
	}
	item = replace(item, " ", "_");
	showParametersDialog(_PT_TRAINING,item);
}


function showParametersDialog(_PT,parameterGroup) {
	requires("1.53d");
	requireInboundPT(_PT);

	if(_PT == _PT_TRAINING){
		dialog_title = "";
		title_split = split(parameterGroup,"_");
		for(i=0;i<title_split.length;i++){
			dialog_title=dialog_title+toUpperCase(substring(title_split[i],0,1))+substring(title_split[i],1)+" ";
		}
		Dialog.create(dialog_title);  
	}else if(_PT == _PT_EVALUATE){
		Dialog.create("Evaluation Parameters"); 
	}else if(_PT == _PT_PREDICT){
		Dialog.create("Prediction Parameters"); 
	}else{
		print("Unable to create Parameters Dialog ! Incorrect type");
		exit();
	}

	readParameters(_PT);
	nb_fields=0;
	for (i = 0; i < _CURRENT_PARAMETER_GROUP.length; i++) {
		if (_CURRENT_PARAMETER_GROUP[i]!=parameterGroup) continue;
		nb_fields++;
		help = replace(_CURRENT_PARAMETER_HELP[i], '\\. ', ".\n");
		if(_CURRENT_PARAMETER_TYPE[i]=="string" || _CURRENT_PARAMETER_TYPE[i]=="file" || _CURRENT_PARAMETER_TYPE[i]=="directory"){
			if(!help.contains("NullHelp")){
				Dialog.addMessage(help);
			}
		}

		if (_CURRENT_PARAMETER_TYPE[i]=="string") {
			Dialog.addString(_CURRENT_PARAMETER_DISPLAY[i], _CURRENT_PARAMETER_VALUE[i], 20);
		}
		if (_CURRENT_PARAMETER_TYPE[i]=="file") {
			Dialog.addFile(_CURRENT_PARAMETER_DISPLAY[i], _CURRENT_PARAMETER_VALUE[i]);
		}
		if (_CURRENT_PARAMETER_TYPE[i]=="directory") {
			Dialog.addDirectory(_CURRENT_PARAMETER_DISPLAY[i], _CURRENT_PARAMETER_VALUE[i]);
		}

		if (_CURRENT_PARAMETER_TYPE[i]=="int" || _CURRENT_PARAMETER_TYPE[i]=="float") {
			if (_CURRENT_PARAMETER_TYPE[i]=="int"){
				Dialog.addNumber(_CURRENT_PARAMETER_DISPLAY[i], _CURRENT_PARAMETER_VALUE[i], 0, 20, '');
			}else{
				Dialog.addNumber(_CURRENT_PARAMETER_DISPLAY[i], _CURRENT_PARAMETER_VALUE[i], 8, 20, '');
			}
		}
		if (_CURRENT_PARAMETER_TYPE[i]=="bool") {
			Dialog.addCheckbox(_CURRENT_PARAMETER_DISPLAY[i], _CURRENT_PARAMETER_VALUE[i]);
		}

		if (_CURRENT_PARAMETER_TYPE[i]=="int" || _CURRENT_PARAMETER_TYPE[i]=="float" || _CURRENT_PARAMETER_TYPE[i]=="bool"){
			if(!help.contains("NullHelp")){
				Dialog.addToSameRow();
				Dialog.addMessage(help);
			}
		}
	}

	if(nb_fields==0){
		Dialog.addMessage("There are no parameters in this category");
	}
	Dialog.show();
	for (i = 0; i < _CURRENT_PARAMETER_GROUP.length; i++) {
		if (_CURRENT_PARAMETER_GROUP[i]!=parameterGroup) continue;
		if (_CURRENT_PARAMETER_TYPE[i]=="string" || _CURRENT_PARAMETER_TYPE[i]=="file" || _CURRENT_PARAMETER_TYPE[i]=="directory") {
			_CURRENT_PARAMETER_VALUE[i] = Dialog.getString();
		}
		if (_CURRENT_PARAMETER_TYPE[i]=="int" || _CURRENT_PARAMETER_TYPE[i]=="float") {
			_CURRENT_PARAMETER_VALUE[i] = Dialog.getNumber();
		}
		if (_CURRENT_PARAMETER_TYPE[i]=="bool") {
			_CURRENT_PARAMETER_VALUE[i] = Dialog.getCheckbox();
		}
	}
	saveParameters(_PT);
}

function showLoadParametersDialog() {
	Dialog.create("Load Parameters");
	Dialog.addFile("Option File","");
	Dialog.show();
	path = Dialog.getString();
	loadParameterFromFile(path);
}

function showInstallEnvDialog() {
	oldEnv = _CONDA_ENV;
	Dialog.create("Install deep-learning env.");
	Dialog.addMessage("Pressing ok, will create a conda environment with the given name\nand install the packages needed by dl4mic in the environment.\nIf an environment with the given name already exists it will be deleted\nand a new one with the same name will be created.\n");
	Dialog.addString("name: ", _CONDA_ENV);
	Dialog.show();
	_CONDA_ENV = Dialog.getString();
	installEnv(oldEnv, _CONDA_ENV);	
}

function showPythonInterpreterDialog() {
	if (_PYTHON_INTERPRETER=='') {
		_PYTHON_INTERPRETER = findPythonInterpreter();
	}
	Dialog.create("Python interpreter");
	Dialog.addFile("Python interpreter: ", _PYTHON_INTERPRETER);
	Dialog.show();
	_PYTHON_INTERPRETER = Dialog.getString();
}

function selectNetwork() {
	setNetwork(getArgument());
}


function setNetwork(network){
   _CURRENT_NETWORK = network;
   _LOADED_PARAMETERS = -1;
   print("DL4Mic - Current network: ", _CURRENT_NETWORK);
}


function askLogToOpen(){
	Dialog.create("Log Opener");
	Dialog.addChoice("Log to Open:",_LOG_FILENAME);
	Dialog.show()
	_pt = -1;
	logFile = Dialog.getChoice();
	if(logFile == _LOG_FILENAME[0]){_pt=0;}
	if(logFile == _LOG_FILENAME[1]){_pt=1;}
	if(logFile == _LOG_FILENAME[2]){_pt=2;}
	
	if(_pt==-1){
		print("Unable to catch log! Incorrect log type");
		exit;
	}
	return _pt;
}


//Miscellaneous

function requireNetwork(){
	if (_CURRENT_NETWORK=='None') {
		showMessage("Please select a network first!");
		exit;
	}
}

function requireInboundPT(_PT){
	if(0 > _PT || _PT >= 3){
		showMessage("Error: Parameter type Out of Bound");
		exit;
	}	
}

function saveParameterString(pt){
	parameters = getParameterString(pt);
	
	baseDir = getValueOfParameter(pt,'baseDir');
	name = getValueOfParameter(pt,'name');
	outPath = baseDir + name + "/options.txt";
	parameters="--network "+_CURRENT_NETWORK+" "+parameters;
	File.saveString(parameters, outPath);
}

function setSubtitles(imageID, columnName, label) {
	selectImage(imageID);
	column = Table.getColumn(columnName);	
	for (i = 0; i < column.length; i++) {
			Property.setSliceLabel(label + " " + column[i],(i+1));
	}
}

function emptyTmpFolder() {
	folder = getDirectory("imagej") + "dl4mic" + File.separator + "tmp";
	inFolder  = folder + File.separator + "in";
	outFolder  = folder + File.separator + "out";
	files = getFileList(inFolder);
	for (i = 0; i < files.length; i++) {
		file = files[i];
		File.delete(inFolder + File.separator + file);
	}
	files = getFileList(outFolder);
	for (i = 0; i < files.length; i++) {
		file = files[i];
		File.delete(outFolder + File.separator + file);
	}
}

function containsFileWithPrefix(folder, prefix) {
	files = getFileList(folder);
	for (i = 0; i < files.length; i++) {
		file = files[i];
		if(startsWith(file, prefix)) {
			return true;
		}
	}
	return false;
}

function openImageSeries(path, name, file, zoomOut) {
	parameter = "open=["+ path + "] sort ";
	if(file!="") {
		parameter = parameter + "filter="+file; 
	}
	run("Image Sequence...", parameter);
	id = getImageID();
	rename(name);
	for (i = 0; i < zoomOut; i++) {
		run("Out [-]");
	}
	return id;
}

function findPythonInterpreter() {
	for(i=0;i<_INTERPRETER_PATH.length;i++){
		if (File.exists(_INTERPRETER_PATH[i])) {
			return _INTERPRETER_PATH[i];
		}
	}
	return '';
}

function initNetworksArray() {
	var _NETWORKS_DIR = getDirectory("imagej") + "dl4mic/networks";
	files = getFileList(_NETWORKS_DIR);
	networks = newArray(0);
	for (i = 0; i < files.length; i++) {
			file = files[i];
			if (File.isDirectory(_NETWORKS_DIR + File.separator+ file) && file!='lib/') {
				name = replace(file, '/', '');
				networks = Array.concat(networks, name);
			}
	}
	return networks;
}

function displayTrainingEvaluationPlot(baseDir,model) {
	path = ""+ baseDir + model + File.separator + "Quality Control" + File.separator + "training_evaluation.csv";
	if(!File.exists(path)){
		print("Unable to find training evaluation file");
		return;
	}
	os = toLowerCase(getInfo("os.name"));
	if (indexOf(os, "win")>-1) {
		text = File.openAsString(path);
		text = replace(text, "\n\n", "\n");
		lines = split(text, "\n");
		
		header = split(lines[0],",");
		loss_id=-1;
		valloss_id = -1;
		for(i=0;i< header.length;i++){
			if(header[i].matches("loss")){
				loss_id=i;
			}if(header[i].matches("val_loss")){
				valloss_id=i;
			}
		}
		if(loss_id==-1){
			print("Loss column not found, taking first column as loss");
			loss_id=0;
		}
		if(valloss_id==-1){
			print("Val Loss column not found, taking second column as val_loss");
			valloss_id=1;
		}
		
		loss = newArray(0);
		valLoss = newArray(0);
		for (i = 1; i < lines.length; i++) {
			line = lines[i];
			parts = split(line, ',');
			if (parts.length<2) continue;
			currentLoss = parseFloat(parts[loss_id]);
			currentValLoss = parseFloat(parts[valloss_id]);
			loss = Array.concat(loss, currentLoss);
			valLoss = Array.concat(valLoss, currentValLoss);
		}
		Table.create(path);
		Table.setColumn("loss", loss, path);
		Table.setColumn("val_loss", valLoss, path);	
		
	} else {
		open(path);
	}
	
	tableTitle = getInfo("window.title");
	loss = Table.getColumn("loss", tableTitle);
	valLoss = Table.getColumn("val_loss", tableTitle);
	xValues = Array.getSequence(loss.length+1);
	xValues = Array.deleteIndex(xValues,0);
	
	Plot.create("training evaluation "+model+"", "epoch", "loss");
	Plot.setColor("orange");
	Plot.setLineWidth(2);
	Plot.add("line", xValues, loss, "training loss");
	Plot.setLineWidth(2);
	Plot.setColor("blue");
	Plot.add("line", xValues, valLoss, "validation loss");
	Plot.setLimitsToFit();
	Plot.setLegend("training loss\tvalidation loss", "top-right");
	Plot.show();
}

function catchLog(_PT){
	requireNetwork();
	requireInboundPT(_PT);
	
	_COMPRESSED_EPOCH_LOG = true;
	_LIVE_TRAINING_PLOT = true;
	
	if (isOpen("Log")) selectWindow("Log");
	logPath = getDirectory("imagej") + "dl4mic"+File.separator +"networks"+File.separator+_CURRENT_NETWORK+File.separator+_LOG_FILENAME[_PT];
	exists = File.exists(logPath);
	print("log path", logPath);
	waiting_points = ".";
	for (i = 0; i < 20; i++) {
		print("\\Update:"+"Searching for Log file"+waiting_points);
		waiting_points=waiting_points+".";
		if (exists) {
			break;
		}
		wait(500);
		exists = File.exists(logPath);
	}
	if(File.exists(logPath)){
		out = File.openAsString(logPath);
	}else{
		print("Unable to find log, aborting macro");
		exit();
	}
	count = 0;
	finished = false;
	endFound = false;

	endString = "";
	if(_PT == _PT_TRAINING){
		endString = "---training done---";
	}else if(_PT == _PT_EVALUATE){
		endString = "---evaluation done---";
	}else if(_PT == _PT_PREDICT){
		endString = "---predictions done---";
	}else{
		print("Unable to catch log! Incorrect type");
		exit();
	}
	
	end   =-1;
	last_a=-1;
	last_b=-1;
	
	loss_value = newArray();
	valloss_value = newArray();
	plot_started = false;
	tableTitle = _CURRENT_NETWORK+" Current Training";
	if(_PT == _PT_TRAINING && _LIVE_TRAINING_PLOT){
		Plot.create(tableTitle, "epoch", "loss");
	}
	while (!finished){
		if (endFound) finished = true;
		lines = split(out, "\n");
		if(lines.length < end){
			print("Log File is shorter than before");
			exit();
		}
		start = count;
		end = lines.length;
	
		
		for (i = start; i < end; i++) {
			printed= false;
			if(_COMPRESSED_EPOCH_LOG){
				curr=split(lines[i]," ");
				if(curr.length>0){
					tmp=split(curr[0],"/");
					if(tmp.length==2){
						a = tmp[0];b = tmp[1];
						if(b==last_b){
							print("\\Update:"+lines[i]);
							printed=true;
						}
						last_a = a;last_b = b;
					}
				}else{
					last_a=-1;last_b=-1;
				}
			}
			
			if(!printed){
				print(lines[i]);
			}
			if(_PT == _PT_TRAINING && _LIVE_TRAINING_PLOT){
				if(lines[i].startsWith("Epoch")){
					if(i==0){
						
					}else{
						epoch_split=split(lines[i-1]," ");
						loss_set = false;
						valloss_set = false;
						for(s=0;s<epoch_split.length;s++){
							if(epoch_split[s]=="loss:"){
								loss_value = Array.concat(loss_value,parseFloat(epoch_split[s+1]));
								loss_set = true;
							}
							if(epoch_split[s]=="val_loss:"){
								valloss_value = Array.concat(valloss_value,parseFloat(epoch_split[s+1]));
								valloss_set = true;
							}
						}
						if(!loss_set){
							loss_value = Array.concat(loss_value,NaN);
						}
						if(!valloss_set){
							valloss_value = Array.concat(valloss_value,NaN);
						}
						xValues = Array.getSequence(loss_value.length);
						if(plot_started){
							Plot.setColor("orange");
							Plot.replace( 0,"line", xValues, loss_value);
							Plot.setColor("blue");
							Plot.replace( 1,"line", xValues, valloss_value);
							Plot.setLegend("training loss\tvalidation loss", "top-right");
							Plot.setStyle(0, "orange,none,2.0,Line");
							Plot.setStyle(1, "blue,none,2.0,Line");
						}else{
							if(loss_value[loss_value.length-1]!=NaN && valloss_value[valloss_value.length-1]!=NaN){
								Plot.setColor("orange");
								Plot.setLineWidth(2);
								Plot.add("line", xValues, loss_value, "training loss");
								Plot.setColor("blue");
								Plot.setLineWidth(2);
								Plot.add("line", xValues, valloss_value, "validation loss");
								Plot.update();
								plot_started=true;
							}
						}
						Plot.setLimits(0,xValues.length+1,NaN,NaN);
						//Plot.setLimitsToFit();
					}
				}
			}
		}
		count=end;
		endFound = indexOf(out, endString)!=-1;
		
		wait(500);
		if (File.exists(logPath)){
		    out = File.openAsString(logPath);
	    }
	}
	if(_PT==_PT_TRAINING && _LIVE_TRAINING_PLOT){
		close(tableTitle);
	}
}

macro "Test MultiLaunch"{
	nbLaunch = 5
	
	name = getValueOfParameter(_PT_TRAINING,'name');
	
	for (l = 0; l < nbLaunch; l++) {
		print("Launching Training "+l);
		readParameters(_PT_TRAINING);
		for (i = 0; i < _CURRENT_PARAMETER_GROUP.length; i++) {
			if (_CURRENT_PARAMETER_NAME[i]=='name') _CURRENT_PARAMETER_VALUE[i] = "M"+l+"_"+name;
		}
		saveParameters(_PT_TRAINING);
		readParameters(_PT_TRAINING);
		train();
		print("Finished Training "+l);
	}
}
