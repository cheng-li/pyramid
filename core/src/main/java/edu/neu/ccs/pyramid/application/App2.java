package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelPredictionAnalysis;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.*;
import edu.neu.ccs.pyramid.multilabel_classification.thresholding.MacroFMeasureTuner;
import edu.neu.ccs.pyramid.multilabel_classification.thresholding.TunedMarginalClassifier;
import edu.neu.ccs.pyramid.optimization.EarlyStopper;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.util.*;
import edu.neu.ccs.pyramid.visualization.Visualizer;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 6/4/17.
 */
public class App2 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        main(config);
    }

    public static void main(Config config) throws Exception{

        Logger logger = Logger.getAnonymousLogger();
        String logFile = config.getString("output.log");
        FileHandler fileHandler = null;
        if (!logFile.isEmpty()){
            new File(logFile).getParentFile().mkdirs();
            //todo should append?
            fileHandler = new FileHandler(logFile, true);
            java.util.logging.Formatter formatter = new SimpleFormatter();
            fileHandler.setFormatter(formatter);
            logger.addHandler(fileHandler);
            logger.setUseParentHandlers(false);
        }


        logger.info(config.toString());

        new File(config.getString("output.folder")).mkdirs();

        if (config.getBoolean("train")){
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            train(config, logger);
            logger.info("total training time = "+stopWatch);

            Config calibrationConfig = new Config();
            calibrationConfig.setString("input.testSet",Paths.get(config.getString("input.folder"), "data_sets", config.getString("input.testData")).toString());
            calibrationConfig.setString("input.validSet",Paths.get(config.getString("input.folder"), "data_sets", config.getString("input.validData")).toString());
            calibrationConfig.setString("input.model", Paths.get(config.getString("output.folder"),"model_app3").toString());
            calibrationConfig.setString("out",config.getString("output.folder"));
            Calibration.main(calibrationConfig, logger);


            if (config.getString("predict.target").equals("macroFMeasure")){
                logger.info("predict.target=macroFMeasure,  user needs to run 'tune' before predictions can be made. " +
                        "Reports will be generated after tuning.");
            } else {
                if (config.getBoolean("train.generateReports")){
                    report(config,config.getString("input.trainData"), logger);
                    if (config.getString("predict.target").equals("subsetAccuracy")){
                        reportCalibrated(config,config.getString("input.trainData"), logger);
                    }
                }

            }
            File metaDataFolder = new File(config.getString("input.folder"),"meta_data");
            config.store(new File(metaDataFolder, "saved_config_app2"));

        }

        if (config.getBoolean("tune")){
            tuneForMacroF(config, logger);
            File metaDataFolder = new File(config.getString("input.folder"),"meta_data");
            Config savedConfig = new Config(new File(metaDataFolder, "saved_config_app2"));
            if (savedConfig.getBoolean("train.generateReports")){
                report(config,config.getString("input.trainData"), logger);
                if (config.getString("predict.target").equals("subsetAccuracy")){
                    reportCalibrated(config,config.getString("input.trainData"), logger);
                }
            }
        }

        if (config.getBoolean("test")){
            report(config,config.getString("input.testData"), logger);
            if (config.getString("predict.target").equals("subsetAccuracy")){
                reportCalibrated(config,config.getString("input.testData"), logger);
            }
        }



        if (fileHandler!=null){
            fileHandler.close();
        }
    }

    static MultiLabelClfDataSet loadData(Config config, String dataName) throws Exception{
        File dataFile = new File(new File(config.getString("input.folder"),
                "data_sets"),dataName);
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(dataFile, DataSetType.ML_CLF_SPARSE,
                true);
        return dataSet;
    }

    static void train(Config config, Logger logger) throws Exception{
        String output = config.getString("output.folder");
        int numIterations = config.getInt("train.numIterations");
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int minDataPerLeaf = config.getInt("train.minDataPerLeaf");
        String modelName = "model_app3";
        int randomSeed = config.getInt("train.randomSeed");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet allTrainData = loadData(config,config.getString("input.trainData"));
        MultiLabelClfDataSet trainSetForEval = minibatch(allTrainData, config.getInt("train.showProgress.sampleSize"),0+randomSeed);

        MultiLabelClfDataSet validSet = loadData(config,config.getString("input.validData"));


        int numClasses = allTrainData.getNumClasses();
        logger.info("number of class = "+numClasses);

        IMLGradientBoosting boosting;
        List<EarlyStopper> earlyStoppers;
        List<Terminator> terminators;
        boolean[] shouldStop;
        int numLabelsLeftToTrain;
        int startIter;
        List<Pair<Integer,Double>> trainingTime;
        List<Pair<Integer,Double>> accuracy;
        double startTime = 0;


        boolean earlyStop = config.getBoolean("train.earlyStop");
        CheckPoint checkPoint;

        if (config.getBoolean("train.warmStart")){
            checkPoint = (CheckPoint) Serialization.deserialize(new File(output, "checkpoint"));
            boosting = checkPoint.boosting;
            earlyStoppers = checkPoint.earlyStoppers;
            terminators = checkPoint.terminators;
            shouldStop = checkPoint.shouldStop;
            numLabelsLeftToTrain = checkPoint.numLabelsLeftToTrain;
            startIter = checkPoint.lastIter+1;
            trainingTime = checkPoint.trainingTime;
            accuracy = checkPoint.accuracy;
            startTime = checkPoint.trainingTime.get(trainingTime.size()-1).getSecond();
        } else {
            boosting  = new IMLGradientBoosting(numClasses);
            earlyStoppers = new ArrayList<>();
            terminators = new ArrayList<>();
            trainingTime = new ArrayList<>();
            accuracy = new ArrayList<>();

            if (earlyStop){
                for (int l=0;l<numClasses;l++){
                    EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MINIMIZE, config.getInt("train.earlyStop.patience"));
                    earlyStopper.setMinimumIterations(config.getInt("train.earlyStop.minIterations"));
                    earlyStoppers.add(earlyStopper);
                }

                for (int l=0;l<numClasses;l++){
                    Terminator terminator = new Terminator();
                    terminator.setMaxStableIterations(config.getInt("train.earlyStop.patience"))
                            .setMinIterations(config.getInt("train.earlyStop.minIterations")/config.getInt("train.showProgress.interval"))
                            .setAbsoluteEpsilon(config.getDouble("train.earlyStop.absoluteChange"))
                            .setRelativeEpsilon(config.getDouble("train.earlyStop.relativeChange"))
                            .setOperation(Terminator.Operation.OR);
                    terminators.add(terminator);
                }
            }
            shouldStop = new boolean[allTrainData.getNumClasses()];
            numLabelsLeftToTrain = numClasses;
            checkPoint = new CheckPoint();
            checkPoint.boosting = boosting;
            checkPoint.earlyStoppers = earlyStoppers;
            checkPoint.terminators = terminators;
            checkPoint.shouldStop = shouldStop;
            // this is not a pointer, has to be updated
            checkPoint.numLabelsLeftToTrain = numLabelsLeftToTrain;
            checkPoint.lastIter = 0;
            checkPoint.trainingTime = trainingTime;
            checkPoint.accuracy = accuracy;
            startIter = 1;
        }
        List<MultiLabel> allAssignments = DataSetUtil.gatherMultiLabels(allTrainData);
        boosting.setAssignments(allAssignments);

        logger.info("During training, the performance is reported using Hamming loss optimal predictor. The performance is computed approximately with "+config.getInt("train.showProgress.sampleSize")+" instances.");

        int progressInterval = config.getInt("train.showProgress.interval");


        int interval = config.getInt("train.fullScanInterval");
        int minibatchLifeSpan = config.getInt("train.minibatchLifeSpan");
        int numActiveFeatures = config.getInt("train.numActiveFeatures");
        int numofLabels = allTrainData.getNumClasses();

        List<Integer>[] activeFeaturesLists = new ArrayList[numofLabels];

        for(int labelnum =0; labelnum<numofLabels; labelnum++){
            activeFeaturesLists[labelnum] = new ArrayList<>();
        }

        MultiLabelClfDataSet trainBatch = null;
        IMLGBTrainer trainer = null;

        StopWatch timeWatch = new StopWatch();
        timeWatch.start();

        for (int i=startIter;i<=numIterations;i++){

            logger.info("iteration "+i);

            if(i%minibatchLifeSpan == 1||i==startIter) {
                trainBatch = minibatch(allTrainData, config.getInt("train.batchSize"),i+randomSeed);
                IMLGBConfig imlgbConfig = new IMLGBConfig.Builder(trainBatch)
                        .learningRate(learningRate)
                        .minDataPerLeaf(minDataPerLeaf)
                        .numLeaves(numLeaves)
                        .numSplitIntervals(config.getInt("train.numSplitIntervals"))
                        .usePrior(config.getBoolean("train.usePrior"))
                        .numActiveFeatures(numActiveFeatures)
                        .build();

                trainer = new IMLGBTrainer(imlgbConfig, boosting, shouldStop);
            }

            if (i % interval == 1) {
                trainer.iterate(activeFeaturesLists, true);
            } else {
                trainer.iterate(activeFeaturesLists, false);
            }


            checkPoint.lastIter+=1;
            if (earlyStop && (i%progressInterval==0 || i==numIterations)){
                for (int l=0;l<numClasses;l++){
                    EarlyStopper earlyStopper = earlyStoppers.get(l);
                    Terminator terminator = terminators.get(l);
                    if (!shouldStop[l]){
                        double kl = KL(boosting, validSet, l);
                        earlyStopper.add(i,kl);
                        terminator.add(kl);
                        if (earlyStopper.shouldStop() || terminator.shouldTerminate()){
                            logger.info("training for label "+l+" ("+allTrainData.getLabelTranslator().toExtLabel(l)+") should stop now");
                            logger.info("the best number of training iterations for the label is "+earlyStopper.getBestIteration());
                            if (i!=earlyStopper.getBestIteration()){
                                boosting.cutTail(l, earlyStopper.getBestIteration());
                                logger.info("roll back the model for this label to iteration "+earlyStopper.getBestIteration());
                            }

                            shouldStop[l]=true;
                            numLabelsLeftToTrain -= 1;
                            checkPoint.numLabelsLeftToTrain = numLabelsLeftToTrain;
                            logger.info("the number of labels left to be trained on = "+numLabelsLeftToTrain);
                        }
                    }
                }

            }

            if (config.getBoolean("train.showTrainProgress") && (i%progressInterval==0 || i==numIterations)){
                logger.info("training set performance (computed approximately with Hamming loss predictor on "+config.getInt("train.showProgress.sampleSize")+" instances).");
                logger.info(new MLMeasures(boosting,trainSetForEval).toString());
            }
            if (config.getBoolean("train.showValidProgress") && (i%progressInterval==0 || i==numIterations)){
                logger.info("validation set performance (computed approximately with Hamming loss predictor)");
                MLMeasures validPerformance = new MLMeasures(boosting,validSet);
                logger.info(validPerformance.toString());
                accuracy.add(new Pair<>(i, validPerformance.getInstanceAverage().getF1()));
            }

            trainingTime.add(new Pair<>(i, startTime+timeWatch.getTime()/1000.0));

            Serialization.serialize(checkPoint, new File(output,"checkpoint"));
            File serializedModel =  new File(output,modelName);
            boosting.serialize(serializedModel);

            if (numLabelsLeftToTrain==0){
                logger.info("all label training finished");
                break;
            }
        }

        logger.info("training done");
        logger.info(stopWatch.toString());

        File outputdir = new File(config.getString("output.folder"));
        outputdir.mkdirs();

        File timeFile = new File(outputdir,"training_time.txt");
        StringBuilder trainTimeBuilder = new StringBuilder();
        for(int i=0;i<trainingTime.size();i++){
            Pair<Integer,Double> timePair = trainingTime.get(i);
            trainTimeBuilder.append("iteration=").append(timePair.getFirst()).append(": ").append(timePair.getSecond()).append("\n");
        }
        FileUtils.writeStringToFile(timeFile,trainTimeBuilder.toString());

        File accuracyFile = new File(outputdir,"valid_instance_f1.txt");
        StringBuilder accuracyBuilder = new StringBuilder();
        for(int i=0;i<accuracy.size();i++){
            Pair<Integer,Double> accuracyPair = accuracy.get(i);
            accuracyBuilder.append("iteration=").append(accuracyPair.getFirst()).append(": ").append(accuracyPair.getSecond()).append("\n");
        }
        FileUtils.writeStringToFile(accuracyFile,accuracyBuilder.toString());

        if (true){
            ObjectMapper objectMapper = new ObjectMapper();
            List<LabelModel> labelModels = IMLGBInspector.getAllRules(boosting);
            new File(output,"decision_rules").mkdirs();

            for (int l=0;l<boosting.getNumClasses();l++){
                objectMapper.writeValue(Paths.get(output, "decision_rules", l+".json").toFile(),labelModels.get(l));
            }

        }

        if (earlyStop){
            for (int l=0;l<numClasses;l++){
                logger.info("----------------------------------------------------");
                logger.info("test performance history for label "+l+": "+earlyStoppers.get(l).history());
                logger.info("model size for label "+l+" = "+(boosting.getRegressors(l).size()-1));
            }
        }

        boolean topFeaturesToFile = true;

        if (topFeaturesToFile){
            logger.info("start writing top features");
            int limit = config.getInt("report.topFeatures.limit");
            List<TopFeatures> topFeaturesList = IntStream.range(0,boosting.getNumClasses())
                    .mapToObj(k -> IMLGBInspector.topFeatures(boosting, k, limit)).collect(Collectors.toList());
            ObjectMapper mapper = new ObjectMapper();
            String file = "top_features.json";
            mapper.writeValue(new File(output,file), topFeaturesList);

            StringBuilder sb = new StringBuilder();
            for (int l=0;l<boosting.getNumClasses();l++){
                sb.append("-------------------------").append("\n");
                sb.append(allTrainData.getLabelTranslator().toExtLabel(l)).append(":").append("\n");
                for (Feature feature: topFeaturesList.get(l).getTopFeatures()){
                    sb.append(feature.simpleString()).append(", ");
                }
                sb.append("\n");
            }
            FileUtils.writeStringToFile(new File(output, "top_features.txt"), sb.toString());

            logger.info("finish writing top features");
        }

    }

    static void tuneForMacroF(Config config, Logger logger) throws Exception{
        logger.info("start tuning for macro F measure");
        String output = config.getString("output.folder");
        String modelName = "model_app3";
        double beta = config.getDouble("tune.FMeasure.beta");

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(output,modelName));;


        MultiLabelClfDataSet dataSet = loadData(config,config.getString("input.validData"));
        double[] thresholds = MacroFMeasureTuner.tuneThresholds(boosting,dataSet,beta);
        TunedMarginalClassifier tunedMarginalClassifier = new TunedMarginalClassifier(boosting,thresholds);
        Serialization.serialize(tunedMarginalClassifier, new File(output,"predictor_macro_f"));
        logger.info("finish tuning for macro F measure");

    }

    static void report(Config config, String dataName, Logger logger) throws Exception{
        logger.info("generating reports for data set "+dataName);
        String output = config.getString("output.folder");
        String modelName = "model_app3";
        File analysisFolder = new File(new File(output,"reports_app3"),dataName+"_reports");
        analysisFolder.mkdirs();
        FileUtils.cleanDirectory(analysisFolder);

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(output,modelName));;

        String predictTarget = config.getString("predict.target");

        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = null;

        switch (predictTarget){
            case "subsetAccuracy":
                pluginPredictorTmp = new SubsetAccPredictor(boosting);
                break;
            case "hammingLoss":
                pluginPredictorTmp = new HammingPredictor(boosting);
                break;
            case "instanceFMeasure":
                pluginPredictorTmp = new InstanceF1Predictor(boosting);
                break;
            case "macroFMeasure":
                TunedMarginalClassifier  tunedMarginalClassifier = (TunedMarginalClassifier)Serialization.deserialize(new File(output, "predictor_macro_f"));
                pluginPredictorTmp = new MacroF1Predictor(boosting,tunedMarginalClassifier);
                break;
            default:
                throw new IllegalArgumentException("unknown prediction target measure "+predictTarget);
        }

        // just to make Lambda expressions happy
        final PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;

        MultiLabelClfDataSet dataSet = loadData(config,dataName);

        MLMeasures mlMeasures = new MLMeasures(pluginPredictor,dataSet);
        mlMeasures.getMacroAverage().setLabelTranslator(dataSet.getLabelTranslator());

        logger.info("performance on dataset "+dataName);
        logger.info(mlMeasures.toString());



        boolean simpleCSV = true;
        if (simpleCSV){
            logger.info("start generating simple CSV report");
            double probThreshold=config.getDouble("report.classProbThreshold");
            File csv = new File(analysisFolder,"report.csv");
            List<Integer> list = IntStream.range(0,dataSet.getNumDataPoints()).boxed().collect(Collectors.toList());
            ParallelStringMapper<Integer> mapper = (list1, i) -> IMLGBInspector.simplePredictionAnalysis(boosting,pluginPredictor,dataSet, list1.get(i),probThreshold);
            ParallelFileWriter.mapToString(mapper,list, csv,100  );
            logger.info("finish generating simple CSV report");
        }





        boolean rulesToJson = config.getBoolean("report.showPredictionDetail");
        if (rulesToJson){
            logger.info("start writing rules to json");
            int ruleLimit = config.getInt("report.rule.limit");
            int numDocsPerFile = config.getInt("report.numDocsPerFile");
            int numFiles = (int)Math.ceil((double)dataSet.getNumDataPoints()/numDocsPerFile);

            double probThreshold=config.getDouble("report.classProbThreshold");
            int labelSetLimit = config.getInt("report.labelSetLimit");


            IntStream.range(0,numFiles).forEach(i->{
                int start = i*numDocsPerFile;
                int end = start+numDocsPerFile;
                List<MultiLabelPredictionAnalysis> partition = IntStream.range(start,Math.min(end,dataSet.getNumDataPoints())).parallel().mapToObj(a->
                        IMLGBInspector.analyzePrediction(boosting, pluginPredictor, dataSet, a,  ruleLimit,labelSetLimit, probThreshold)).collect(Collectors.toList());
                ObjectMapper mapper = new ObjectMapper();

                String file = "report_"+(i+1)+".json";
                try {
                    mapper.writeValue(new File(analysisFolder,file), partition);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                logger.info("progress = "+ Progress.percentage(i+1,numFiles));
            });

            logger.info("finish writing rules to json");
        }


        boolean dataInfoToJson = true;
        if (dataInfoToJson){
            logger.info("start writing data info to json");
            Set<String> modelLabels = IntStream.range(0,boosting.getNumClasses()).mapToObj(i->boosting.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            Set<String> dataSetLabels = DataSetUtil.gatherLabels(dataSet).stream().map(i -> dataSet.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            JsonGenerator jsonGenerator = new JsonFactory().createGenerator(new File(analysisFolder,"data_info.json"), JsonEncoding.UTF8);
            jsonGenerator.writeStartObject();
            jsonGenerator.writeStringField("dataSet",dataName);
            jsonGenerator.writeNumberField("numClassesInModel",boosting.getNumClasses());
            jsonGenerator.writeNumberField("numClassesInDataSet",dataSetLabels.size());
            jsonGenerator.writeNumberField("numClassesInModelDataSetCombined",dataSet.getNumClasses());
            Set<String> modelNotDataLabels = SetUtil.complement(modelLabels, dataSetLabels);
            Set<String> dataNotModelLabels = SetUtil.complement(dataSetLabels,modelLabels);
            jsonGenerator.writeNumberField("numClassesInDataSetButNotModel",dataNotModelLabels.size());
            jsonGenerator.writeNumberField("numClassesInModelButNotDataSet",modelNotDataLabels.size());
            jsonGenerator.writeArrayFieldStart("classesInDataSetButNotModel");
            for (String label: dataNotModelLabels){
                jsonGenerator.writeObject(label);
            }
            jsonGenerator.writeEndArray();
            jsonGenerator.writeArrayFieldStart("classesInModelButNotDataSet");
            for (String label: modelNotDataLabels){
                jsonGenerator.writeObject(label);
            }
            jsonGenerator.writeEndArray();
            jsonGenerator.writeNumberField("labelCardinality",dataSet.labelCardinality());

            jsonGenerator.writeEndObject();
            jsonGenerator.close();
            logger.info("finish writing data info to json");
        }


        boolean modelConfigToJson = true;
        if (modelConfigToJson){
            logger.info("start writing model config to json");
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"model_config.json"),config);
            logger.info("finish writing model config to json");
        }

        boolean dataConfigToJson = true;
        if (dataConfigToJson){
            logger.info("start writing data config to json");
            File dataConfigFile = Paths.get(config.getString("input.folder"),
                    "data_sets",dataName,"data_config.json").toFile();
            if (dataConfigFile.exists()){
                FileUtils.copyFileToDirectory(dataConfigFile,analysisFolder);
            }
            logger.info("finish writing data config to json");
        }

        boolean performanceToJson = true;
        if (performanceToJson){
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"performance.json"),mlMeasures);
        }

        boolean individualPerformance = true;
        if (individualPerformance){
            logger.info("start writing individual label performance to json");
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"individual_performance.json"),mlMeasures.getMacroAverage());
            logger.info("finish writing individual label performance to json");
        }

        if (config.getBoolean("report.produceHTML")){
            logger.info("start producing html files");

            Config savedApp1Config = new Config(Paths.get(config.getString("input.folder"), "meta_data","saved_config_app1").toFile());

            List<String> hosts = savedApp1Config.getStrings("index.hosts");
            List<Integer> ports = savedApp1Config.getIntegers("index.ports");

            //todo make it better
            if (savedApp1Config.getString("index.clientType").equals("node")){
                hosts = new ArrayList<>();
                for (int port: ports){
                    hosts.add("localhost");
                }
                //default setting
                hosts.add("localhost");
                ports.add(9200);
            }
            try (Visualizer visualizer = new Visualizer(logger, hosts, ports)){
                visualizer.produceHtml(analysisFolder);
                logger.info("finish producing html files");
            }


        }

        logger.info("reports generated");
    }


    static void reportCalibrated(Config config, String dataName, Logger logger) throws Exception{
        logger.info("generating reports with calibrated probabilities for data set "+dataName);
        String output = config.getString("output.folder");
        String modelName = "model_app3";
        String setCalibration = "set_calibration";
        String labelCalibration = "label_calibration";
        File analysisFolder = new File(new File(output,"reports_app3"),dataName+"_reports_calibrated");
        analysisFolder.mkdirs();
        FileUtils.cleanDirectory(analysisFolder);

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(output,modelName));
        IMLGBIsotonicScaling setScaling = (IMLGBIsotonicScaling)Serialization.deserialize(new File(output,setCalibration));
        IMLGBLabelIsotonicScaling labelScaling = (IMLGBLabelIsotonicScaling)Serialization.deserialize(new File(output, labelCalibration));

        String predictTarget = config.getString("predict.target");

        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = null;

        switch (predictTarget){
            case "subsetAccuracy":
                pluginPredictorTmp = new SubsetAccPredictor(boosting);
                break;
            case "hammingLoss":
                pluginPredictorTmp = new HammingPredictor(boosting);
                break;
            case "instanceFMeasure":
                pluginPredictorTmp = new InstanceF1Predictor(boosting);
                break;
            case "macroFMeasure":
                TunedMarginalClassifier  tunedMarginalClassifier = (TunedMarginalClassifier)Serialization.deserialize(new File(output, "predictor_macro_f"));
                pluginPredictorTmp = new MacroF1Predictor(boosting,tunedMarginalClassifier);
                break;
            default:
                throw new IllegalArgumentException("unknown prediction target measure "+predictTarget);
        }

        // just to make Lambda expressions happy
        final PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;

        MultiLabelClfDataSet dataSet = loadData(config,dataName);

        logger.info("sum of calibrated probabilities");

        double[] all = IntStream.range(0, dataSet.getNumDataPoints())
                .mapToDouble(dataPointIndex-> Arrays.stream(boosting.predictAllAssignmentProbsWithConstraint(dataSet.getRow(dataPointIndex)))
                .map(setScaling::calibratedProb).sum()).toArray();
        DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics(all);
        logger.info(descriptiveStatistics.toString());

        MLMeasures mlMeasures = new MLMeasures(pluginPredictor,dataSet);
        mlMeasures.getMacroAverage().setLabelTranslator(dataSet.getLabelTranslator());

        logger.info("performance on dataset "+dataName);
        logger.info(mlMeasures.toString());

        List<Integer> reportIdOrderTmp = IntStream.range(0,dataSet.getNumDataPoints()).boxed().collect(Collectors.toList());
        if (config.getString("report.order").equals("confidence")){
            Comparator<Pair<Integer,Double>> confidenceComparator = Comparator.comparing(pair->pair.getSecond());
                    reportIdOrderTmp =
                    IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToObj(i->{
                        MultiLabel prediction = pluginPredictor.predict(dataSet.getRow(i));
                        double confidence = setScaling.calibratedProb(dataSet.getRow(i), prediction);
                        return new Pair<>(i, confidence);
                    }).sorted(confidenceComparator.reversed())
                            .map(Pair::getFirst)
                            .collect(Collectors.toList());
        }

        if (config.getString("report.order").equals("mistake")){
            Comparator<Pair<Integer,Double>> mistakeComparator = Comparator.comparing(pair->pair.getSecond());
            reportIdOrderTmp =
                    IntStream.range(0,dataSet.getNumDataPoints()).parallel().mapToObj(i->{
                        MultiLabel prediction = pluginPredictor.predict(dataSet.getRow(i));
                        double confidence = setScaling.calibratedProb(dataSet.getRow(i), prediction);
                        double instanceF1 = FMeasure.f1(prediction,dataSet.getMultiLabels()[i]);
                        return new Pair<>(i, (1-instanceF1)*confidence);
                    }).sorted(mistakeComparator.reversed())
                            .map(Pair::getFirst)
                            .collect(Collectors.toList());
        }

        // just to please lambda expression
        final List<Integer> reportIdOrder = reportIdOrderTmp;

        boolean simpleCSV = true;
        if (simpleCSV){
            logger.info("start generating simple CSV report");
            double probThreshold=config.getDouble("report.classProbThreshold");
            File csv = new File(analysisFolder,"report.csv");
            ParallelStringMapper<Integer> mapper = new ParallelStringMapper<Integer>() {
                @Override
                public String mapToString(List<Integer> list, int i) {
                    return IMLGBInspector.simplePredictionAnalysisCalibrated(boosting, setScaling,labelScaling, pluginPredictor,dataSet,list.get(i),probThreshold);

                }
            };
            ParallelFileWriter.mapToString(mapper, reportIdOrder,csv,100);
            logger.info("finish generating simple CSV report");
        }





        boolean rulesToJson = config.getBoolean("report.showPredictionDetail");
        if (rulesToJson){
            logger.info("start writing rules to json");
            int ruleLimit = config.getInt("report.rule.limit");
            int numDocsPerFile = config.getInt("report.numDocsPerFile");
            int numFiles = (int)Math.ceil((double)dataSet.getNumDataPoints()/numDocsPerFile);

            double probThreshold=config.getDouble("report.classProbThreshold");
            int labelSetLimit = config.getInt("report.labelSetLimit");


            IntStream.range(0,numFiles).forEach(i->{
                int start = i*numDocsPerFile;
                int end = start+numDocsPerFile;
                List<MultiLabelPredictionAnalysis> partition = IntStream.range(start,Math.min(end,dataSet.getNumDataPoints())).parallel().mapToObj(a->
                        IMLGBInspector.analyzePredictionCalibrated(boosting, setScaling, labelScaling,pluginPredictor, dataSet, reportIdOrder.get(a),  ruleLimit,labelSetLimit, probThreshold)).collect(Collectors.toList());
                ObjectMapper mapper = new ObjectMapper();

                String file = "report_"+(i+1)+".json";
                try {
                    mapper.writeValue(new File(analysisFolder,file), partition);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                logger.info("progress = "+ Progress.percentage(i+1,numFiles));
            });

            logger.info("finish writing rules to json");
        }


        boolean dataInfoToJson = true;
        if (dataInfoToJson){
            logger.info("start writing data info to json");
            Set<String> modelLabels = IntStream.range(0,boosting.getNumClasses()).mapToObj(i->boosting.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            Set<String> dataSetLabels = DataSetUtil.gatherLabels(dataSet).stream().map(i -> dataSet.getLabelTranslator().toExtLabel(i))
                    .collect(Collectors.toSet());

            JsonGenerator jsonGenerator = new JsonFactory().createGenerator(new File(analysisFolder,"data_info.json"), JsonEncoding.UTF8);
            jsonGenerator.writeStartObject();
            jsonGenerator.writeStringField("dataSet",dataName);
            jsonGenerator.writeNumberField("numClassesInModel",boosting.getNumClasses());
            jsonGenerator.writeNumberField("numClassesInDataSet",dataSetLabels.size());
            jsonGenerator.writeNumberField("numClassesInModelDataSetCombined",dataSet.getNumClasses());
            Set<String> modelNotDataLabels = SetUtil.complement(modelLabels, dataSetLabels);
            Set<String> dataNotModelLabels = SetUtil.complement(dataSetLabels,modelLabels);
            jsonGenerator.writeNumberField("numClassesInDataSetButNotModel",dataNotModelLabels.size());
            jsonGenerator.writeNumberField("numClassesInModelButNotDataSet",modelNotDataLabels.size());
            jsonGenerator.writeArrayFieldStart("classesInDataSetButNotModel");
            for (String label: dataNotModelLabels){
                jsonGenerator.writeObject(label);
            }
            jsonGenerator.writeEndArray();
            jsonGenerator.writeArrayFieldStart("classesInModelButNotDataSet");
            for (String label: modelNotDataLabels){
                jsonGenerator.writeObject(label);
            }
            jsonGenerator.writeEndArray();
            jsonGenerator.writeNumberField("labelCardinality",dataSet.labelCardinality());

            jsonGenerator.writeEndObject();
            jsonGenerator.close();
            logger.info("finish writing data info to json");
        }


        boolean modelConfigToJson = true;
        if (modelConfigToJson){
            logger.info("start writing model config to json");
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"model_config.json"),config);
            logger.info("finish writing model config to json");
        }

        boolean dataConfigToJson = true;
        if (dataConfigToJson){
            logger.info("start writing data config to json");
            File dataConfigFile = Paths.get(config.getString("input.folder"),
                    "data_sets",dataName,"data_config.json").toFile();
            if (dataConfigFile.exists()){
                FileUtils.copyFileToDirectory(dataConfigFile,analysisFolder);
            }
            logger.info("finish writing data config to json");
        }

        boolean performanceToJson = true;
        if (performanceToJson){
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"performance.json"),mlMeasures);
        }

        boolean individualPerformance = true;
        if (individualPerformance){
            logger.info("start writing individual label performance to json");
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(new File(analysisFolder,"individual_performance.json"),mlMeasures.getMacroAverage());
            logger.info("finish writing individual label performance to json");
        }

        if (config.getBoolean("report.produceHTML")){
            logger.info("start producing html files");

            Config savedApp1Config = new Config(Paths.get(config.getString("input.folder"), "meta_data","saved_config_app1").toFile());

            List<String> hosts = savedApp1Config.getStrings("index.hosts");
            List<Integer> ports = savedApp1Config.getIntegers("index.ports");

            //todo make it better
            if (savedApp1Config.getString("index.clientType").equals("node")){
                hosts = new ArrayList<>();
                for (int port: ports){
                    hosts.add("localhost");
                }
                //default setting
                hosts.add("localhost");
                ports.add(9200);
            }
            try (Visualizer visualizer = new Visualizer(logger, hosts, ports)){
                visualizer.produceHtml(analysisFolder);
                logger.info("finish producing html files");
            }


        }

        logger.info("reports generated");
    }



    private static double KL(IMLGradientBoosting boosting, Vector vector, MultiLabel multiLabel, int classIndex){
        double[] p = new double[2];
        if (multiLabel.matchClass(classIndex)){
            p[0] = 0;
            p[1] = 1;
        } else {
            p[0] = 1;
            p[1] = 0;
        }
        double[] logQ = boosting.predictLogClassProbs(vector, classIndex);
        return KLDivergence.klGivenPLogQ(p, logQ);
    }

    private static double KL(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, int classIndex){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->KL(boosting, dataSet.getRow(i), dataSet.getMultiLabels()[i], classIndex))
                .average().getAsDouble();
    }



    private static MultiLabelClfDataSet minibatch(MultiLabelClfDataSet allData, int minibatchSize, int interation){
        List<Integer> all = IntStream.range(0, allData.getNumDataPoints()).boxed().collect(Collectors.toList());
        Collections.shuffle(all, new Random(interation));
        List<Integer> keep = all.stream().limit(minibatchSize).collect(Collectors.toList());
        return DataSetUtil.sampleData(allData, keep);
    }

    public static class CheckPoint implements Serializable{
        private static final long serialVersionUID = 2L;
        private IMLGradientBoosting boosting;
        private List<EarlyStopper> earlyStoppers;
        private List<Terminator> terminators;
        private boolean[] shouldStop;
        private int numLabelsLeftToTrain;
        private int lastIter;
        private List<Pair<Integer,Double>> trainingTime;
        private List<Pair<Integer,Double>> accuracy;

        public int getLastIter() {
            return lastIter;
        }
    }
}
