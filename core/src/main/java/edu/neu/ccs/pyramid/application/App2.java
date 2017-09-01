package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
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
import edu.neu.ccs.pyramid.util.Progress;
import edu.neu.ccs.pyramid.util.Serialization;
import edu.neu.ccs.pyramid.util.SetUtil;
import edu.neu.ccs.pyramid.visualization.Visualizer;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
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

            if (config.getString("predict.target").equals("macroFMeasure")){
                logger.info("predict.target=macroFMeasure,  user needs to run 'tune' before predictions can be made. " +
                        "Reports will be generated after tuning.");
            } else {
                if (config.getBoolean("train.generateReports")){
                    report(config,config.getString("input.trainData"), logger);
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
            }
        }

        if (config.getBoolean("test")){
            report(config,config.getString("input.testData"), logger);
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

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet allTrainData = loadData(config,config.getString("input.trainData"));
        MultiLabelClfDataSet trainSetForEval = minibatch(allTrainData, config.getInt("train.showProgress.sampleSize"));


        MultiLabelClfDataSet testSetForEval = null;
        if (config.getBoolean("train.showTestProgress") || config.getBoolean("train.earlyStop")){
            MultiLabelClfDataSet testSet = loadData(config,config.getString("input.testData"));
            testSetForEval = minibatch(testSet, config.getInt("train.showProgress.sampleSize"));
        }

        int numClasses = allTrainData.getNumClasses();
        logger.info("number of class = "+numClasses);


        IMLGradientBoosting boosting;
        if (config.getBoolean("train.warmStart")){
            boosting = IMLGradientBoosting.deserialize(new File(output,modelName));
        } else {
            boosting  = new IMLGradientBoosting(numClasses);
        }
        List<MultiLabel> allAssignments = DataSetUtil.gatherMultiLabels(allTrainData);
        boosting.setAssignments(allAssignments);

        logger.info("During training, the performance is reported using Hamming loss optimal predictor. The performance is computed approximately with "+config.getInt("train.showProgress.sampleSize")+" instances.");



        boolean earlyStop = config.getBoolean("train.earlyStop");

        List<EarlyStopper> earlyStoppers = new ArrayList<>();
        List<Terminator> terminators = new ArrayList<>();
        boolean[] shouldStop = new boolean[allTrainData.getNumClasses()];

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


        int numLabelsLeftToTrain = numClasses;

        int progressInterval = config.getInt("train.showProgress.interval");
        for (int i=1;i<=numIterations;i++){
            logger.info("iteration "+i);
            MultiLabelClfDataSet trainBatch = minibatch(allTrainData, config.getInt("train.batchSize"));

            IMLGBConfig imlgbConfig = new IMLGBConfig.Builder(trainBatch)
                    .learningRate(learningRate)
                    .minDataPerLeaf(minDataPerLeaf)
                    .numLeaves(numLeaves)
                    .numSplitIntervals(config.getInt("train.numSplitIntervals"))
                    .usePrior(config.getBoolean("train.usePrior"))
                    .featureSamplingRate(config.getDouble("train.featureSamplingRate"))
                    .build();

            IMLGBTrainer trainer = new IMLGBTrainer(imlgbConfig,boosting, shouldStop);
            trainer.iterateWithoutStagingScores();
            if (earlyStop && (i%progressInterval==0 || i==numIterations)){
                for (int l=0;l<numClasses;l++){
                    EarlyStopper earlyStopper = earlyStoppers.get(l);
                    Terminator terminator = terminators.get(l);
                    if (!shouldStop[l]){
                        double kl = KL(boosting, testSetForEval, l);
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
                            logger.info("the number of labels left to be trained on = "+numLabelsLeftToTrain);
                        }
                    }
                }
                File serializedModel =  new File(output,modelName);
                //todo pick best models

                boosting.serialize(serializedModel);
            }
            if (config.getBoolean("train.showTrainProgress") && (i%progressInterval==0 || i==numIterations)){
                logger.info("training set performance (computed approximately with Hamming loss predictor on "+config.getInt("train.showProgress.sampleSize")+" instances).");
                logger.info(new MLMeasures(boosting,trainSetForEval).toString());
            }
            if (config.getBoolean("train.showTestProgress") && (i%progressInterval==0 || i==numIterations)){
                logger.info("test set performance (computed approximately with Hamming loss predictor on "+config.getInt("train.showProgress.sampleSize")+" instances).");
                logger.info(new MLMeasures(boosting,testSetForEval).toString());
            }
            if (numLabelsLeftToTrain==0){
                logger.info("all label training finished");
                break;
            }
        }
        logger.info("training done");
        File serializedModel =  new File(output,modelName);
        //todo pick best models

        boosting.serialize(serializedModel);
        logger.info(stopWatch.toString());

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
        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(output,modelName));
        String tuneBy = config.getString("tune.data");
        String dataName;
        switch (tuneBy){
            case "train":
                dataName = config.getString("input.trainData");
                break;
            case "test":
                dataName = config.getString("input.testData");
                break;
            default:
                throw new IllegalArgumentException("tune.data should be train or test");
        }


        MultiLabelClfDataSet dataSet = loadData(config,dataName);
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

        IMLGradientBoosting boosting = IMLGradientBoosting.deserialize(new File(output,modelName));
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
        mlMeasures.getMacroAverage().setLabelTranslator(boosting.getLabelTranslator());

        logger.info("performance on dataset "+dataName);
        logger.info(mlMeasures.toString());

        boolean simpleCSV = true;
        if (simpleCSV){
            logger.info("start generating simple CSV report");
            double probThreshold=config.getDouble("report.classProbThreshold");
            File csv = new File(analysisFolder,"report.csv");
            List<String> strs = IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                    .mapToObj(i->IMLGBInspector.simplePredictionAnalysis(boosting,pluginPredictor,dataSet,i,probThreshold))
                    .collect(Collectors.toList());
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(csv))){
                for (int i=0;i<dataSet.getNumDataPoints();i++){
                    String str = strs.get(i);
                    bw.write(str);
                }
            }
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

    private static MultiLabelClfDataSet minibatch(MultiLabelClfDataSet allData, int minibatchSize){
        List<Integer> all = IntStream.range(0, allData.getNumDataPoints()).boxed().collect(Collectors.toList());
        Collections.shuffle(all);
        List<Integer> keep = all.stream().limit(minibatchSize).collect(Collectors.toList());
        return DataSetUtil.sampleData(allData, keep);
    }
}
