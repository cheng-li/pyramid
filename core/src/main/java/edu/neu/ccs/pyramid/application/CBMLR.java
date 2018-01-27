package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.AveragePrecision;
import edu.neu.ccs.pyramid.eval.LogLikelihood;
import edu.neu.ccs.pyramid.eval.MAP;
import edu.neu.ccs.pyramid.eval.MLMeasures;
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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * CBM with L2 regularized LR base learners
 * Created by chengli on 4/2/17.
 */
public class CBMLR {
    private static boolean VERBOSE = false;

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        VERBOSE = config.getBoolean("output.verbose");

        new File(config.getString("output.dir")).mkdirs();

        if (config.getBoolean("tune")){
            System.out.println("============================================================");
            System.out.println("Start hyper parameter tuning");
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            List<TuneResult> tuneResults = new ArrayList<>();
            List<MultiLabelClfDataSet> dataSets = loadTrainValidData(config);
            List<Double> variances = config.getDoubles("tune.variance.candidates");
            List<Integer> components = config.getIntegers("tune.numComponents.candidates");
            for (double variance: variances){
                for (int component: components){
                    StopWatch stopWatch1 = new StopWatch();
                    stopWatch1.start();
                    HyperParameters hyperParameters = new HyperParameters();
                    hyperParameters.numComponents = component;
                    hyperParameters.variance = variance;
                    System.out.println("---------------------------");
                    System.out.println("Trying hyper parameters:");
                    System.out.println("train.numComponents = "+hyperParameters.numComponents);
                    System.out.println("train.variance = "+hyperParameters.variance);
                    TuneResult tuneResult = tune(config, hyperParameters, dataSets.get(0), dataSets.get(1));
                    System.out.println("Found optimal train.iterations = "+tuneResult.hyperParameters.iterations);
                    System.out.println("Validation performance = "+tuneResult.performance);
                    tuneResults.add(tuneResult);
                    System.out.println("Time spent on trying this set of hyper parameters = "+stopWatch1);
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
                case "instance_log_likelihood":
                    best = tuneResults.stream().max(comparator).get();
                    break;
                default:
                    throw new IllegalArgumentException("tune.targetMetric should be instance_set_accuracy, instance_f1, instance_hamming_loss, instance_log_likelihood");
            }


            System.out.println("---------------------------");
            System.out.println("Hyper parameter tuning done.");
            System.out.println("Time spent on entire hyper parameter tuning = "+stopWatch);
            System.out.println("Best validation performance = "+best.performance);
            System.out.println("Best hyper parameters:");
            System.out.println("train.numComponents = "+best.hyperParameters.numComponents);
            System.out.println("train.variance = "+best.hyperParameters.variance);
            System.out.println("train.iterations = "+best.hyperParameters.iterations);
            Config tunedHypers = best.hyperParameters.asConfig();
            tunedHypers.store(new File(config.getString("output.dir"), "tuned_hyper_parameters.properties"));
            System.out.println("Tuned hyper parameters saved to "+new File(config.getString("output.dir"), "tuned_hyper_parameters.properties").getAbsolutePath());
            System.out.println("============================================================");
        }

        if (config.getBoolean("train")){
            System.out.println("============================================================");
            if (config.getBoolean("train.useTunedHyperParameters")){
                File hyperFile = new File(config.getString("output.dir"), "tuned_hyper_parameters.properties");
                if (!hyperFile.exists()){
                    System.out.println("train.useTunedHyperParameters is set to true. But no tuned hyper parameters can be found in the output directory.");
                    System.out.println("Please either run hyper parameter tuning, or provide hyper parameters manually and set train.useTunedHyperParameters=false.");
                    System.exit(1);
                }
                Config tunedHypers = new Config(hyperFile);
                HyperParameters hyperParameters = new HyperParameters(tunedHypers);
                System.out.println("Start training with tuned hyper parameters:");
                System.out.println("train.numComponents = "+hyperParameters.numComponents);
                System.out.println("train.variance = "+hyperParameters.variance);
                System.out.println("train.iterations = "+hyperParameters.iterations);


                MultiLabelClfDataSet trainSet = loadTrainData(config);
                train(config, hyperParameters, trainSet);
            } else {
                HyperParameters hyperParameters = new HyperParameters(config);
                System.out.println("Start training with given hyper parameters:");
                System.out.println("train.numComponents = "+hyperParameters.numComponents);
                System.out.println("train.variance = "+hyperParameters.variance);
                System.out.println("train.iterations = "+hyperParameters.iterations);

                MultiLabelClfDataSet trainSet = loadTrainData(config);
                train(config, hyperParameters, trainSet);
            }
            System.out.println("============================================================");
        }

        if (config.getBoolean("test")){
            System.out.println("============================================================");
            test(config);
            System.out.println("============================================================");
        }

    }


    private static TuneResult tune(Config config, HyperParameters hyperParameters, MultiLabelClfDataSet trainSet, MultiLabelClfDataSet validSet) throws Exception{
        List<Integer> unobservedLabels = DataSetUtil.unobservedLabels(trainSet);
        CBM cbm = newCBM(config,trainSet, hyperParameters);
        EarlyStopper earlyStopper = loadNewEarlyStopper(config);

        LRCBMOptimizer optimizer = getOptimizer(config, hyperParameters, cbm, trainSet);
        optimizer.initialize();

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
            case "instance_log_likelihood":
                // acc predictor seems to be the best match for log likelihood
                AccPredictor accPredictor1 = new AccPredictor(cbm);
                accPredictor1.setComponentContributionThreshold(config.getDouble("predict.piThreshold"));
                classifier = accPredictor1;
                break;
            default:
                throw new IllegalArgumentException("predictTarget should be instance_set_accuracy, instance_f1 or instance_hamming_loss");
        }

        int interval = config.getInt("tune.monitorInterval");

        for (int iter = 1; true; iter++){

            if (VERBOSE){
                System.out.println("iteration "+iter );
            }


            optimizer.iterate();

            if (iter%interval==0){


                MLMeasures validMeasures = new MLMeasures(classifier,validSet);
                if (VERBOSE){
                    System.out.println("validation performance with "+predictTarget+" optimal predictor:");
                    System.out.println(validMeasures);
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
                    case "instance_log_likelihood":
                        earlyStopper.add(iter, LogLikelihood.averageLogLikelihood(cbm, validSet, unobservedLabels));
                        break;
                    default:
                        throw new IllegalArgumentException("predictTarget should be instance_set_accuracy, instance_f1 or instance_hamming_loss");
                }

                if (earlyStopper.shouldStop()){
                    if (VERBOSE){
                        System.out.println("Early Stopper: the training should stop now!");
                    }

                    break;
                }
            }
        }

        if (VERBOSE){
            System.out.println("done!");
        }

        hyperParameters.iterations = earlyStopper.getBestIteration();
        TuneResult tuneResult = new TuneResult();
        tuneResult.hyperParameters = hyperParameters;
        tuneResult.performance = earlyStopper.getBestValue();
        return tuneResult;

    }


    private static void train(Config config, HyperParameters hyperParameters, MultiLabelClfDataSet trainSet) throws Exception{

        List<Integer> unobservedLabels = DataSetUtil.unobservedLabels(trainSet);

        if (!unobservedLabels.isEmpty()){
            System.out.println("The following labels do not actually appear in the training set and therefore cannot be learned:");
            System.out.println(ListUtil.toSimpleString(unobservedLabels));
        }
        String output = config.getString("output.dir");
        FileUtils.writeStringToFile(new File(output,"unobserved_labels.txt"), ListUtil.toSimpleString(unobservedLabels));

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        CBM cbm = newCBM(config,trainSet, hyperParameters);

        LRCBMOptimizer optimizer = getOptimizer(config, hyperParameters, cbm, trainSet);
        System.out.println("Initializing the model");
        optimizer.initialize();
        System.out.println("Initialization done");

        for (int iter=1;iter<=hyperParameters.iterations;iter++){
            System.out.println("Training progress: iteration "+iter );
            optimizer.iterate();
        }

        System.out.println("training done!");
        System.out.println("time spent on training = "+stopWatch);

        Serialization.serialize(cbm, new File(output,"model"));
        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(trainSet);
        Serialization.serialize(support, new File(output,"support"));



        System.out.println();

        System.out.println("Making predictions on train set with 3 different predictors designed for different metrics:");
        reportAccPrediction(config, cbm, trainSet, "train");
        reportF1Prediction(config, cbm, trainSet, "train");
        reportHammingPrediction(config, cbm, trainSet, "train");
        reportGeneral(config, cbm, trainSet, "train");
        System.out.println();
    }

    private static void test(Config config) throws Exception{
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSetAutoSparseSequential(config.getString("input.testData"));

        String output = config.getString("output.dir");

        CBM cbm = (CBM) Serialization.deserialize(new File(output, "model"));


        System.out.println();

        System.out.println("Making predictions on test set with 3 different predictors designed for different metrics:");
        reportAccPrediction(config, cbm, testSet, "test");
        reportF1Prediction(config, cbm, testSet, "test");
        reportHammingPrediction(config, cbm, testSet, "test");
        reportGeneral(config, cbm, testSet, "test");
        System.out.println();
    }

    private static void reportAccPrediction(Config config, CBM cbm, MultiLabelClfDataSet dataSet, String name) throws Exception{
        System.out.println("============================================================");
        System.out.println("Making predictions on "+name +" set with the instance set accuracy optimal predictor");
        String output = config.getString("output.dir");
        AccPredictor accPredictor = new AccPredictor(cbm);
        accPredictor.setComponentContributionThreshold(config.getDouble("predict.piThreshold"));
        MultiLabel[] predictions = accPredictor.predict(dataSet);
        MLMeasures mlMeasures = new MLMeasures(dataSet.getNumClasses(),dataSet.getMultiLabels(),predictions);
        System.out.println(name+" performance with the instance set accuracy optimal predictor");
        System.out.println(mlMeasures);
        File performanceFile = Paths.get(output,name+"_predictions", "instance_accuracy_optimal","performance.txt").toFile();
        FileUtils.writeStringToFile(performanceFile, mlMeasures.toString());
        System.out.println(name+" performance is saved to "+performanceFile.toString());


        // Here we do not use approximation
        double[] setProbs = IntStream.range(0, predictions.length).parallel().
                mapToDouble(i->cbm.predictAssignmentProb(dataSet.getRow(i),predictions[i])).toArray();
        File predictionFile = Paths.get(output,name+"_predictions", "instance_accuracy_optimal","predictions.txt").toFile();
        try (BufferedWriter br = new BufferedWriter(new FileWriter(predictionFile))){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                br.write(predictions[i].toString());
                br.write(":");
                br.write(""+setProbs[i]);
                br.newLine();
            }
        }

        System.out.println("predicted sets and their probabilities are saved to "+predictionFile.getAbsolutePath());
        System.out.println("============================================================");
    }


    private static void reportF1Prediction(Config config, CBM cbm, MultiLabelClfDataSet dataSet, String name) throws Exception{
        System.out.println("============================================================");
        System.out.println("Making predictions on "+name+" set with the instance F1 optimal predictor");
        String output = config.getString("output.dir");
        PluginF1 pluginF1 = new PluginF1(cbm);
        List<MultiLabel> support = (List<MultiLabel>) Serialization.deserialize(new File(output, "support"));
        pluginF1.setSupport(support);
        pluginF1.setPiThreshold(config.getDouble("predict.piThreshold"));
        MultiLabel[] predictions = pluginF1.predict(dataSet);
        MLMeasures mlMeasures = new MLMeasures(dataSet.getNumClasses(),dataSet.getMultiLabels(),predictions);
        System.out.println(name+" performance with the instance F1 optimal predictor");
        System.out.println(mlMeasures);
        File performanceFile = Paths.get(output,name+"_predictions", "instance_f1_optimal","performance.txt").toFile();
        FileUtils.writeStringToFile(performanceFile, mlMeasures.toString());
        System.out.println(name+" performance is saved to "+performanceFile.toString());


        // Here we do not use approximation
        double[] setProbs = IntStream.range(0, predictions.length).parallel().
                mapToDouble(i->cbm.predictAssignmentProb(dataSet.getRow(i),predictions[i])).toArray();
        File predictionFile = Paths.get(output,name+"_predictions", "instance_f1_optimal","predictions.txt").toFile();
        try (BufferedWriter br = new BufferedWriter(new FileWriter(predictionFile))){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                br.write(predictions[i].toString());
                br.write(":");
                br.write(""+setProbs[i]);
                br.newLine();
            }
        }

        System.out.println("predicted sets and their probabilities are saved to "+predictionFile.getAbsolutePath());
        System.out.println("============================================================");
    }

    private static void reportHammingPrediction(Config config, CBM cbm, MultiLabelClfDataSet dataSet, String name) throws Exception{
        System.out.println("============================================================");
        System.out.println("Making predictions on "+name+" set with the instance Hamming loss optimal predictor");
        String output = config.getString("output.dir");
        MarginalPredictor marginalPredictor = new MarginalPredictor(cbm);
        marginalPredictor.setPiThreshold(config.getDouble("predict.piThreshold"));
        MultiLabel[] predictions = marginalPredictor.predict(dataSet);
        MLMeasures mlMeasures = new MLMeasures(dataSet.getNumClasses(),dataSet.getMultiLabels(),predictions);
        System.out.println(name+" performance with the instance Hamming loss optimal predictor");
        System.out.println(mlMeasures);
        File performanceFile = Paths.get(output,name+"_predictions", "instance_hamming_loss_optimal","performance.txt").toFile();
        FileUtils.writeStringToFile(performanceFile, mlMeasures.toString());
        System.out.println(name+" performance is saved to "+performanceFile.toString());


        // Here we do not use approximation
        double[] setProbs = IntStream.range(0, predictions.length).parallel().
                mapToDouble(i->cbm.predictAssignmentProb(dataSet.getRow(i),predictions[i])).toArray();
        File predictionFile = Paths.get(output,name+"_predictions", "instance_hamming_loss_optimal","predictions.txt").toFile();
        try (BufferedWriter br = new BufferedWriter(new FileWriter(predictionFile))){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                br.write(predictions[i].toString());
                br.write(":");
                br.write(""+setProbs[i]);
                br.newLine();
            }
        }

        System.out.println("predicted sets and their probabilities are saved to "+predictionFile.getAbsolutePath());
        System.out.println("============================================================");
    }


    private static void reportGeneral(Config config, CBM cbm, MultiLabelClfDataSet dataSet, String name) throws Exception{
        System.out.println("============================================================");
        System.out.println("computing other predictor-independent metrics");

        System.out.println("label averaged MAP");
        System.out.println(MAP.map(cbm, dataSet));
        System.out.println("instance averaged MAP");
        System.out.println(MAP.instanceMAP(cbm, dataSet));
        System.out.println("global AP truncated at 30");
        System.out.println(AveragePrecision.globalAveragePrecisionTruncated(cbm, dataSet, 30));

        String output = config.getString("output.dir");
        File labelProbFile = Paths.get(output, name+"_predictions",  "label_probabilities.txt").toFile();
        double labelProbThreshold = config.getDouble("report.labelProbThreshold");

        try (BufferedWriter br = new BufferedWriter(new FileWriter(labelProbFile))){
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                br.write(CBMInspector.topLabels(cbm, dataSet.getRow(i), labelProbThreshold));
                br.newLine();
            }
        }

        System.out.println("individual label probabilities are saved to "+labelProbFile.getAbsolutePath());

        List<Integer> unobservedLabels = Arrays.stream(FileUtils.readFileToString(new File(output,"unobserved_labels.txt"))
                .split(",")).map(s->s.trim()).filter(s->!s.isEmpty()).map(s->Integer.parseInt(s)).collect(Collectors.toList());


        // Here we do not use approximation
        double[] logLikelihoods = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->cbm.predictLogAssignmentProb(dataSet.getRow(i),dataSet.getMultiLabels()[i]))
                .toArray();

        double average = IntStream.range(0, dataSet.getNumDataPoints()).filter(i->!containsNovelClass(dataSet.getMultiLabels()[i],unobservedLabels))
                .mapToDouble(i->logLikelihoods[i]).average().getAsDouble();


        File logLikelihoodFile = Paths.get(output, name+"_predictions", "ground_truth_log_likelihood.txt").toFile();
        FileUtils.writeStringToFile(logLikelihoodFile, PrintUtil.toMutipleLines(logLikelihoods));
        System.out.println("individual log likelihood of the "+name +" ground truth label set is written to "+logLikelihoodFile.getAbsolutePath());

        System.out.println("average log likelihood of the "+name+" ground truth label sets = "+average);
        if (!unobservedLabels.isEmpty()&&name.equals("test")){
            System.out.println("This is computed by ignoring test instances with new labels unobserved during training");
            System.out.println("The following labels do not actually appear in the training set and therefore cannot be learned:");
            System.out.println(ListUtil.toSimpleString(unobservedLabels));
        }
    }




    private static LRCBMOptimizer getOptimizer(Config config, HyperParameters hyperParameters, CBM cbm, MultiLabelClfDataSet trainSet){
        LRCBMOptimizer lrcbmOptimizer = new LRCBMOptimizer(cbm, trainSet);
        lrcbmOptimizer.setPriorVarianceBinary(hyperParameters.variance);
        lrcbmOptimizer.setPriorVarianceMultiClass(hyperParameters.variance);
        lrcbmOptimizer.setBinaryUpdatesPerIter(config.getInt("train.updatesPerIteration"));
        lrcbmOptimizer.setMulticlassUpdatesPerIter(config.getInt("train.updatesPerIteration"));
        lrcbmOptimizer.setSkipDataThreshold(config.getDouble("train.skipDataThreshold"));
        lrcbmOptimizer.setSkipLabelThreshold(config.getDouble("train.skipLabelThreshold"));
        lrcbmOptimizer.setSmoothingStrength(config.getDouble("train.smoothStrength"));

        return lrcbmOptimizer;
    }


    private static CBM newCBM(Config config, MultiLabelClfDataSet trainSet, HyperParameters hyperParameters){

        CBM cbm;


        cbm = CBM.getBuilder()
                .setNumClasses(trainSet.getNumClasses())
                .setNumFeatures(trainSet.getNumFeatures())
                .setNumComponents(hyperParameters.numComponents)
                .setMultiClassClassifierType("lr")
                .setBinaryClassifierType("lr")
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
                        System.out.println("training set contains empty labels, automatically set predict.allowEmpty = true");
                    }

                } else {
                    cbm.setAllowEmpty(false);
                    if (VERBOSE){
                        System.out.println("training set does not contain empty labels, automatically set predict.allowEmpty = false");
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
            case "instance_log_likelihood":
                earlyStopGoal = EarlyStopper.Goal.MAXIMIZE;
                break;
            default:
                throw new IllegalArgumentException("unsupported tune.targetMetric "+earlyStopMetric);
        }

        EarlyStopper earlyStopper = new EarlyStopper(earlyStopGoal,patience);
        earlyStopper.setMinimumIterations(config.getInt("tune.earlyStop.minIterations"));
        return earlyStopper;
    }

    private static List<MultiLabelClfDataSet> loadTrainValidData(Config config) throws Exception{
        String validPath = config.getString("input.validData");
        List<MultiLabelClfDataSet> datasets = new ArrayList<>();
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSetAutoSparseSequential(config.getString("input.trainData"));

        if (validPath.isEmpty()){
            System.out.println("No external validation data is provided. Use random 20% of the training data for validation.");
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
        String validPath = config.getString("input.validData");
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSetAutoSparseSequential(config.getString("input.trainData"));

        if (validPath.isEmpty()||!config.getBoolean("train.useValidData")){
            return trainSet;
        } else {
            MultiLabelClfDataSet validSet = TRECFormat.loadMultiLabelClfDataSetAutoSparseSequential(config.getString("input.validData"));
            return DataSetUtil.concatenateByRow(trainSet, validSet);
        }

    }

    private static class HyperParameters{
        double variance;
        int iterations;
        int numComponents;

        HyperParameters() {
        }

        HyperParameters(Config config) {
            variance = config.getDouble("train.variance");
            iterations = config.getInt("train.iterations");
            numComponents = config.getInt("train.numComponents");
        }

        Config asConfig(){
            Config config = new Config();
            config.setDouble("train.variance", variance);
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
