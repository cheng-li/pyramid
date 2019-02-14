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
import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class BRLREN {
    private static boolean VERBOSE = false;

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


//        logger.info(config.toString());

        VERBOSE = config.getBoolean("output.verbose");

        new File(config.getString("output.dir")).mkdirs();



        if (config.getBoolean("train")){
            logger.info("============================================================");
            HyperParameters hyperParameters = new HyperParameters(config);


            MultiLabelClfDataSet trainSet = loadTrainData(config);
            MultiLabelClfDataSet validSet = loadValidData(config);
            train(config, hyperParameters, trainSet, validSet, logger);
            logger.info("============================================================");
        }


        if (fileHandler!=null){
            fileHandler.close();
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        main(config);
    }



    private static void train(Config config, HyperParameters hyperParameters, MultiLabelClfDataSet trainSet, MultiLabelClfDataSet validSet,
                              Logger logger) throws Exception{

        List<Integer> unobservedLabels = DataSetUtil.unobservedLabels(trainSet);

        if (!unobservedLabels.isEmpty()){
            logger.info("The following labels do not actually appear in the training set and therefore cannot be learned:");
            logger.info(ListUtil.toSimpleString(unobservedLabels));
            FileUtils.writeStringToFile(Paths.get(config.getString("output.dir"),"model_predictions",config.getString("output.modelFolder"),"analysis","unobserved_labels.txt").toFile(), ListUtil.toSimpleString(unobservedLabels));
        }
        String output = config.getString("output.dir");
        EarlyStopper earlyStopper = loadNewEarlyStopper();

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


        AccPredictor accPredictor = new AccPredictor(cbm);
        accPredictor.setComponentContributionThreshold(config.getDouble("predict.piThreshold"));
        CBM bestModel = null;

        int interval = 1;
        for (int iter=1;true;iter++){

            logger.info("Training progress: iteration "+iter );
            optimizer.iterate();

            if (iter%interval==0){
                MLMeasures validMeasures = new MLMeasures(accPredictor,validSet);
                if (VERBOSE){
                    logger.info("validation performance");
                    logger.info(validMeasures.toString());
                }
                earlyStopper.add(iter,validMeasures.getInstanceAverage().getAccuracy());
                if (earlyStopper.getBestIteration()==iter){
                    bestModel = (CBM)Serialization.deepCopy(cbm);
                }


                if (earlyStopper.shouldStop()){
                    if (VERBOSE){
                        logger.info("Early Stopper: the training should stop now!");
                        logger.info("Early Stopper: best iteration found = "+earlyStopper.getBestIteration());
                        logger.info("Early Stopper: best validation performance = "+earlyStopper.getBestValue());
                    }

                    break;
                }
            }
        }

        logger.info("training done!");
        logger.info("time spent on training = "+stopWatch);

        Serialization.serialize(bestModel, Paths.get(output,"model_predictions",config.getString("output.modelFolder"),"models","classifier"));
        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(trainSet);
        Serialization.serialize(support, Paths.get(output,"model_predictions",config.getString("output.modelFolder"),"models","support"));

        featureImportance(config, bestModel, trainSet.getFeatureList(), trainSet.getLabelTranslator(),logger);
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
            sbcount.append(mlLabelTranslator.toExtLabel(l)).append(":").append(featuresByEach[l]).append("\n");
        }

        String output = config.getString("output.dir");
        Paths.get(output, "model_predictions",config.getString("output.modelFolder"),"analysis").toFile().mkdirs();
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(Paths.get(output, "model_predictions",config.getString("output.modelFolder"),"analysis","top_features.txt").toFile()))){
            for (int l=0;l<cbm.getNumClasses();l++){
                if (cbm.getBinaryClassifiers()[0][l] instanceof LogisticRegression){
                    LogisticRegression logisticRegression = (LogisticRegression) cbm.getBinaryClassifiers()[0][l];
                    logisticRegression.setFeatureList(featureList);
                    List<String> labels = new ArrayList<>();
                    labels.add("not_"+mlLabelTranslator.toExtLabel(l));
                    labels.add(mlLabelTranslator.toExtLabel(l));
                    LabelTranslator labelTranslator = new LabelTranslator(labels);
                    logisticRegression.setLabelTranslator(labelTranslator);
                    TopFeatures topFeatures = LogisticRegressionInspector.topFeatures(logisticRegression, 1,Integer.MAX_VALUE);
                    bw.write("label "+l+" ("+mlLabelTranslator.toExtLabel(l)+")");
                    bw.write(": ");
                    for (int f=0;f<topFeatures.getTopFeatures().size();f++){
                        Feature feature = topFeatures.getTopFeatures().get(f);
                        double utility = topFeatures.getUtilities().get(f);
                        if (utility!=0){
                            bw.write(""+feature.getIndex());
//                        bw.write(" (");
//                        bw.write(feature.getName());
//                        bw.write(")");
                            bw.write(":");
                            bw.write(""+utility);
                            bw.write(", ");
                        }

                    }
                    bw.write("\n");
                }
            }
        }
        logger.info("feature count in each label is saved to the file "+Paths.get(output, "model_predictions",config.getString("output.modelFolder"),"analysis","feature_count_in_each_label.txt").toFile().getAbsolutePath());
        FileUtils.writeStringToFile(Paths.get(output, "model_predictions",config.getString("output.modelFolder"),"analysis","feature_count_in_each_label.txt").toFile(),sbcount.toString());

    }

    private static ENCBMOptimizer getOptimizer(Config config, HyperParameters hyperParameters, CBM cbm, MultiLabelClfDataSet trainSet){
        ENCBMOptimizer optimizer = new ENCBMOptimizer(cbm, trainSet);

        if (config.getBoolean("train.useInstanceWeights")){
            double[] instanceWeights = loadInstanceWeights(config);
            optimizer.setInstanceWeights(instanceWeights);
        }

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
        optimizer.setMaxNumLinearRegUpdates(config.getInt("train.maxNumLinearRegUpdates"));
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
        cbm.setLabelTranslator(trainSet.getLabelTranslator());

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


    private static EarlyStopper loadNewEarlyStopper(){
        int patience = 5;
        EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MAXIMIZE,patience);
        earlyStopper.setMinimumIterations(5);
        return earlyStopper;
    }


    private static MultiLabelClfDataSet loadValidData(Config config) throws Exception{
        MultiLabelClfDataSet validSet = TRECFormat.loadMultiLabelClfDataSetAutoSparseSequential(config.getString("input.validData"));
        return validSet;
    }

    private static MultiLabelClfDataSet loadTrainData(Config config) throws Exception{
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSetAutoSparseSequential(config.getString("input.trainData"));
        return trainSet;
    }

    private static double[] loadInstanceWeights(Config config){
        File file = new File(config.getString("input.trainData"),"instance_weights.txt");
        double[] weights = new double[0];
        try {
            weights = FileUtils.readLines(file).stream().mapToDouble(Double::parseDouble).toArray();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return weights;
    }

    private static class HyperParameters{
        double penalty;
        double l1Ratio;
        int numComponents;

        HyperParameters() {
        }

        HyperParameters(Config config) {
            penalty = config.getDouble("train.penalty");
            l1Ratio = config.getDouble("train.l1Ratio");
            numComponents = config.getInt("train.numComponents");
        }

        Config asConfig(){
            Config config = new Config();
            config.setDouble("train.penalty", penalty);
            config.setDouble("train.l1Ratio", l1Ratio);
            config.setInt("train.numComponents", numComponents);
            return config;
        }


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
