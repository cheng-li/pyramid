package edu.neu.ccs.pyramid.application;


import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.MAP;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.*;
import edu.neu.ccs.pyramid.optimization.EarlyStopper;
import edu.neu.ccs.pyramid.util.ListUtil;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.PrintUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class BRLREN {
    private static boolean VERBOSE = false;

    public static void main(String[] args) throws Exception {

        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }
        
        
        Config config = new Config(args[0]);


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

        VERBOSE = config.getBoolean("output.verbose");

        new File(config.getString("output.dir")).mkdirs();

        if (config.getBoolean("tune")){
            logger.info("============================================================");
            logger.info("Start hyper parameter tuning");
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            List<TuneResult> tuneResults = new ArrayList<>();
            List<MultiLabelClfDataSet> dataSets = loadTrainValidData(config, logger);
            List<Double> penalties = config.getDoubles("tune.penalty.candidates");
            List<Double> l1Ratioes = config.getDoubles("tune.l1Ratio.candidates");
            List<Integer> components = config.getIntegers("tune.numComponents.candidates");
            for (double penalty: penalties){
                for (double l1Ratio : l1Ratioes) {
                    for (int component: components){
                        StopWatch stopWatch1 = new StopWatch();
                        stopWatch1.start();
                        HyperParameters hyperParameters = new HyperParameters();
                        hyperParameters.numComponents = component;
                        hyperParameters.l1Ratio = l1Ratio;
                        hyperParameters.penalty = penalty;
                        logger.info("---------------------------");
                        logger.info("Trying hyper parameters:");
                        logger.info("train.numComponents = "+hyperParameters.numComponents);
                        logger.info("train.penalty = "+hyperParameters.penalty);
                        logger.info("train.l1Ratio = "+hyperParameters.l1Ratio);
                        TuneResult tuneResult = tune(config, hyperParameters, dataSets.get(0), dataSets.get(1), logger);
                        logger.info("Found optimal train.iterations = "+tuneResult.hyperParameters.iterations);
                        logger.info("Validation performance = "+tuneResult.performance);
                        tuneResults.add(tuneResult);
                        logger.info("Time spent on trying this set of hyper parameters = "+stopWatch1);
                    }
                }
            }

            Comparator<TuneResult> comparator = Comparator.comparing(res->res.performance);

            TuneResult best;
            String predictTarget = config.getString("tune.targetMetric");
            switch (predictTarget){
                case "instance_set_accuracy":
                    best = tuneResults.stream().max(comparator).get();
                    break;
                case "instance_f1":
                    best = tuneResults.stream().max(comparator).get();
                    break;
                case "instance_hamming_loss":
                    best = tuneResults.stream().min(comparator).get();
                    break;
                case "label_map":
                    best = tuneResults.stream().max(comparator).get();
                    break;
                default:
                    throw new IllegalArgumentException("tune.targetMetric should be instance_set_accuracy, instance_f1 or instance_hamming_loss");
            }


            logger.info("---------------------------");
            logger.info("Hyper parameter tuning done.");
            logger.info("Time spent on entire hyper parameter tuning = "+stopWatch);
            logger.info("Best validation performance = "+best.performance);
            logger.info("Best hyper parameters:");
            logger.info("train.numComponents = "+best.hyperParameters.numComponents);
            logger.info("train.penalty = "+best.hyperParameters.penalty);
            logger.info("train.l1Ratio = "+best.hyperParameters.l1Ratio);
            logger.info("train.iterations = "+best.hyperParameters.iterations);
            Config tunedHypers = best.hyperParameters.asConfig();
            tunedHypers.store(new File(config.getString("output.dir"), "tuned_hyper_parameters.properties"));
            logger.info("Tuned hyper parameters saved to "+new File(config.getString("output.dir"), "tuned_hyper_parameters.properties").getAbsolutePath());
            logger.info("============================================================");
        }

        if (config.getBoolean("train")){
            logger.info("============================================================");
            if (config.getBoolean("train.useTunedHyperParameters")){
                File hyperFile = new File(config.getString("output.dir"), "tuned_hyper_parameters.properties");
                if (!hyperFile.exists()){
                    logger.info("train.useTunedHyperParameters is set to true. But no tuned hyper parameters can be found in the output directory.");
                    logger.info("Please either run hyper parameter tuning, or provide hyper parameters manually and set train.useTunedHyperParameters=false.");
                    System.exit(1);
                }
                Config tunedHypers = new Config(hyperFile);
                HyperParameters hyperParameters = new HyperParameters(tunedHypers);
                logger.info("Start training with tuned hyper parameters:");
                logger.info("train.numComponents = "+hyperParameters.numComponents);
                logger.info("train.penalty = "+hyperParameters.penalty);
                logger.info("train.l1Ratio = "+hyperParameters.l1Ratio);
                logger.info("train.iterations = "+hyperParameters.iterations);


                MultiLabelClfDataSet trainSet = loadTrainData(config);
                train(config, hyperParameters, trainSet, logger);
            } else {
                HyperParameters hyperParameters = new HyperParameters(config);
                logger.info("Start training with given hyper parameters:");
                logger.info("train.numComponents = "+hyperParameters.numComponents);
                logger.info("train.penalty = "+hyperParameters.penalty);
                logger.info("train.l1Ratio = "+hyperParameters.l1Ratio);
                logger.info("train.iterations = "+hyperParameters.iterations);

                MultiLabelClfDataSet trainSet = loadTrainData(config);
                train(config, hyperParameters, trainSet, logger);
            }
            logger.info("============================================================");
        }


        if (fileHandler!=null){
            fileHandler.close();
        }

    }


    private static TuneResult tune(Config config, HyperParameters hyperParameters, MultiLabelClfDataSet trainSet, MultiLabelClfDataSet validSet,
                                   Logger logger) throws Exception{

        CBM cbm = newCBM(config, trainSet, hyperParameters, logger);
        EarlyStopper earlyStopper = loadNewEarlyStopper(config);

        ENCBMOptimizer optimizer = getOptimizer(config, hyperParameters, cbm, trainSet);
        if (config.getBoolean("train.randomInitialize")) {
            optimizer.randInitialize();
        } else {
            optimizer.initialize();
        }

        MultiLabelClassifier classifier;
        String predictTarget = config.getString("tune.targetMetric");
        switch (predictTarget){
            case "instance_set_accuracy":
                AccPredictor accPredictor = new AccPredictor(cbm);
                accPredictor.setComponentContributionThreshold(config.getDouble("predict.piThreshold"));
                classifier = accPredictor;
                break;
            case "instance_f1":
                PluginF1 pluginF1 = new PluginF1(cbm);
                List<MultiLabel> support = DataSetUtil.gatherMultiLabels(trainSet);
                pluginF1.setSupport(support);
                pluginF1.setPiThreshold(config.getDouble("predict.piThreshold"));
                classifier = pluginF1;
                break;
            case "instance_hamming_loss":
                MarginalPredictor marginalPredictor = new MarginalPredictor(cbm);
                marginalPredictor.setPiThreshold(config.getDouble("predict.piThreshold"));
                classifier = marginalPredictor;
                break;

            case "label_map":
                AccPredictor accPredictor2 = new AccPredictor(cbm);
                accPredictor2.setComponentContributionThreshold(config.getDouble("predict.piThreshold"));
                classifier = accPredictor2;
                break;
            default:
                throw new IllegalArgumentException("predictTarget should be instance_set_accuracy, instance_f1 or instance_hamming_loss");
        }

        int interval = config.getInt("tune.monitorInterval");

        for (int iter = 1; true; iter++){

            if (VERBOSE){
                logger.info("iteration "+iter );
            }


            optimizer.iterate();

            if (iter%interval==0){


                MLMeasures validMeasures = new MLMeasures(classifier,validSet);
                if (VERBOSE){
                    logger.info("validation performance with "+predictTarget+" optimal predictor:");
                    logger.info(validMeasures.toString());
                }

                switch (predictTarget){
                    case "instance_set_accuracy":
                        earlyStopper.add(iter,validMeasures.getInstanceAverage().getAccuracy());
                        break;
                    case "instance_f1":
                        earlyStopper.add(iter,validMeasures.getInstanceAverage().getF1());
                        break;
                    case "instance_hamming_loss":
                        earlyStopper.add(iter,validMeasures.getInstanceAverage().getHammingLoss());
                        break;
                    case "label_map":
                        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(trainSet);
                        double map = MAP.mapBySupport(cbm, validSet,support);
                        earlyStopper.add(iter,map);
                        break;
                    default:
                        throw new IllegalArgumentException("predictTarget should be instance_set_accuracy or instance_f1");
                }

                if (earlyStopper.shouldStop()){
                    if (VERBOSE){
                        logger.info("Early Stopper: the training should stop now!");
                    }

                    break;
                }
            }
        }

        if (VERBOSE){
            logger.info("done!");
        }

        hyperParameters.iterations = earlyStopper.getBestIteration();
        TuneResult tuneResult = new TuneResult();
        tuneResult.hyperParameters = hyperParameters;
        tuneResult.performance = earlyStopper.getBestValue();
        return tuneResult;

    }


    private static void train(Config config, HyperParameters hyperParameters, MultiLabelClfDataSet trainSet,
                              Logger logger) throws Exception{

        List<Integer> unobservedLabels = DataSetUtil.unobservedLabels(trainSet);

        if (!unobservedLabels.isEmpty()){
            logger.info("The following labels do not actually appear in the training set and therefore cannot be learned:");
            logger.info(ListUtil.toSimpleString(unobservedLabels));
        }
        String output = config.getString("output.dir");
        FileUtils.writeStringToFile(new File(output,"unobserved_labels.txt"), ListUtil.toSimpleString(unobservedLabels));

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        CBM cbm = newCBM(config,trainSet, hyperParameters, logger);

        ENCBMOptimizer optimizer = getOptimizer(config, hyperParameters, cbm, trainSet);
        logger.info("Initializing the model");
        if (config.getBoolean("train.randomInitialize")) {
            optimizer.randInitialize();
        } else {
            optimizer.initialize();
        }
        logger.info("Initialization done");

        for (int iter=1;iter<=hyperParameters.iterations;iter++){
            logger.info("Training progress: iteration "+iter );
            optimizer.iterate();
        }

        logger.info("training done!");
        logger.info("time spent on training = "+stopWatch);

        Serialization.serialize(cbm, new File(output,"model"));
        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(trainSet);
        Serialization.serialize(support, new File(output,"support"));

        featureImportance(config, cbm, trainSet.getFeatureList(), trainSet.getLabelTranslator(),logger);

        logger.info("Making predictions on train set with 3 different predictors designed for different metrics:");

    }





    //todo currently only for br
    private static void featureImportance(Config config, CBM cbm, FeatureList featureList, LabelTranslator mlLabelTranslator,
                                          Logger logger) throws Exception{

        logger.info("number of selected features in all labels (union)= "+CBMInspector.usedFeatures(cbm).size());
        int[] featuresByEach = CBMInspector.usedFeaturesByEachLabel(cbm);
        double average = Arrays.stream(featuresByEach).average().getAsDouble();
        logger.info("average number of selected features in each label ="+average);

        StringBuilder sbcount = new StringBuilder();
        for (int l=0;l<featuresByEach.length;l++){
            sbcount.append(cbm.getLabelTranslator().toExtLabel(l)).append(":").append(featuresByEach[l]).append("\n");
        }

        String output = config.getString("output.dir");
        StringBuilder stringBuilder = new StringBuilder();
        for (int l=0;l<cbm.getNumClasses();l++){
            if (cbm.getBinaryClassifiers()[0][l] instanceof LogisticRegression){
                LogisticRegression logisticRegression = (LogisticRegression) cbm.getBinaryClassifiers()[0][l];
                logisticRegression.setFeatureList(featureList);
                List<String> labels = new ArrayList<>();
                labels.add("not_"+mlLabelTranslator.toExtLabel(l));
                labels.add(mlLabelTranslator.toExtLabel(l));
                LabelTranslator labelTranslator = new LabelTranslator(labels);
                logisticRegression.setLabelTranslator(labelTranslator);
                TopFeatures topFeatures = LogisticRegressionInspector.topFeatures(logisticRegression, 1,100);
                stringBuilder.append("label "+l+" ("+mlLabelTranslator.toExtLabel(l)+")").append(": ");
                for (int f=0;f<topFeatures.getTopFeatures().size();f++){
                    Feature feature = topFeatures.getTopFeatures().get(f);
                    double utility = topFeatures.getUtilities().get(f);
                    stringBuilder.append(feature.getIndex()).append(" (").append(feature.getName()).append(")")
                            .append(":").append(utility)
                            .append(", ");
                }
                stringBuilder.append("\n");
            }

        }
        FileUtils.writeStringToFile(new File(output,"top_features.txt"),stringBuilder.toString());
        logger.info("feature count in each label is saved to the file "+new File(output,"feature_count_in_each_label.txt").getAbsolutePath());
        FileUtils.writeStringToFile(new File(output,"feature_count_in_each_label.txt"),sbcount.toString());

    }

    private static ENCBMOptimizer getOptimizer(Config config, HyperParameters hyperParameters, CBM cbm, MultiLabelClfDataSet trainSet){
        ENCBMOptimizer optimizer = new ENCBMOptimizer(cbm, trainSet);

        optimizer.setLineSearch(config.getBoolean("train.elasticnet.lineSearch"));
        optimizer.setRegularizationBinary(hyperParameters.penalty);
        optimizer.setRegularizationMultiClass(hyperParameters.penalty);
        optimizer.setL1RatioBinary(hyperParameters.l1Ratio);
        optimizer.setL1RatioMultiClass(hyperParameters.l1Ratio);
        optimizer.setActiveSet(config.getBoolean("train.elasticnet.activeSet"));

        optimizer.setBinaryUpdatesPerIter(config.getInt("train.updatesPerIteration"));
        optimizer.setMulticlassUpdatesPerIter(config.getInt("train.updatesPerIteration"));
        optimizer.setSkipDataThreshold(config.getDouble("train.skipDataThreshold"));
        optimizer.setSkipLabelThreshold(config.getDouble("train.skipLabelThreshold"));
//

        return optimizer;
    }


    private static CBM newCBM(Config config, MultiLabelClfDataSet trainSet, HyperParameters hyperParameters,
                              Logger logger){

        CBM cbm;


        cbm = CBM.getBuilder()
                .setNumClasses(trainSet.getNumClasses())
                .setNumFeatures(trainSet.getNumFeatures())
                .setNumComponents(hyperParameters.numComponents)
                .setMultiClassClassifierType("elasticnet")
                .setBinaryClassifierType("elasticnet")
                .setDense(true)
                .build();

        String allowEmpty = config.getString("predict.allowEmpty");
        switch (allowEmpty){
            case "true":
                cbm.setAllowEmpty(true);
                break;
            case "false":
                cbm.setAllowEmpty(false);
                break;
            case "auto":
                Set<MultiLabel> seen = DataSetUtil.gatherMultiLabels(trainSet).stream().collect(Collectors.toSet());
                MultiLabel empty = new MultiLabel();
                if (seen.contains(empty)){
                    cbm.setAllowEmpty(true);
                    if (VERBOSE){
                        logger.info("training set contains empty labels, automatically set predict.allowEmpty = true");
                    }

                } else {
                    cbm.setAllowEmpty(false);
                    if (VERBOSE){
                        logger.info("training set does not contain empty labels, automatically set predict.allowEmpty = false");
                    }
                }
                break;
            default:
                throw new IllegalArgumentException("unknown value for predict.allowEmpty");
        }

        return cbm;


    }



    private static EarlyStopper loadNewEarlyStopper(Config config){
        String earlyStopMetric = config.getString("tune.targetMetric");
        int patience = config.getInt("tune.earlyStop.patience");
        EarlyStopper.Goal earlyStopGoal = null;
        switch (earlyStopMetric){
            case "instance_set_accuracy":
                earlyStopGoal = EarlyStopper.Goal.MAXIMIZE;
                break;
            case "instance_f1":
                earlyStopGoal = EarlyStopper.Goal.MAXIMIZE;
                break;
            case "instance_hamming_loss":
                earlyStopGoal = EarlyStopper.Goal.MINIMIZE;
                break;
            case "label_map":
                earlyStopGoal = EarlyStopper.Goal.MAXIMIZE;
                break;
            default:
                throw new IllegalArgumentException("unsupported tune.targetMetric "+earlyStopMetric);
        }

        EarlyStopper earlyStopper = new EarlyStopper(earlyStopGoal,patience);
        earlyStopper.setMinimumIterations(config.getInt("tune.earlyStop.minIterations"));
        return earlyStopper;
    }

    private static List<MultiLabelClfDataSet> loadTrainValidData(Config config, Logger logger) throws Exception{
        String validPath = config.getString("input.validData");
        List<MultiLabelClfDataSet> datasets = new ArrayList<>();
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSetAutoSparseSequential(config.getString("input.trainData"));

        if (validPath.isEmpty()){
            logger.info("No external validation data is provided. Use random 20% of the training data for validation.");
            Pair<MultiLabelClfDataSet, MultiLabelClfDataSet> dataSetPair = DataSetUtil.splitToTrainValidation(trainSet,0.8);
            MultiLabelClfDataSet subTrain = dataSetPair.getFirst();
            MultiLabelClfDataSet validSet  = dataSetPair.getSecond();
            datasets.add(subTrain);
            datasets.add(validSet);
        } else {
            MultiLabelClfDataSet validSet = TRECFormat.loadMultiLabelClfDataSetAutoSparseSequential(config.getString("input.validData"));
            datasets.add(trainSet);
            datasets.add(validSet);
        }
        return datasets;
    }

    private static MultiLabelClfDataSet loadTrainData(Config config) throws Exception{
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSetAutoSparseSequential(config.getString("input.trainData"));
        return trainSet;


    }

    private static class HyperParameters{
        double penalty;
        double l1Ratio;
        int iterations;
        int numComponents;

        HyperParameters() {
        }

        HyperParameters(Config config) {
            penalty = config.getDouble("train.penalty");
            l1Ratio = config.getDouble("train.l1Ratio");
            iterations = config.getInt("train.iterations");
            numComponents = config.getInt("train.numComponents");
        }

        Config asConfig(){
            Config config = new Config();
            config.setDouble("train.penalty", penalty);
            config.setDouble("train.l1Ratio", l1Ratio);
            config.setInt("train.iterations", iterations);
            config.setInt("train.numComponents", numComponents);
            return config;
        }


    }

    private static class TuneResult{
        HyperParameters hyperParameters;
        double performance;
    }

    private static boolean containsNovelClass(MultiLabel multiLabel, List<Integer> novelLabels){
        for (int l:novelLabels){
            if (multiLabel.matchClass(l)){
                return true;
            }
        }
        return false;
    }
}
