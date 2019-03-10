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
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
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
public class BRGB {
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



        new File(config.getString("output.folder")).mkdirs();

        if (config.getBoolean("train")){
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            train(config, logger);
            logger.info("total training time = "+stopWatch);
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

    static void train(Config config, Logger logger) throws Exception{
        String output = config.getString("output.folder");
        int numIterations = config.getInt("train.numIterations");
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int minDataPerLeaf = config.getInt("train.minDataPerLeaf");

        int randomSeed = config.getInt("train.randomSeed");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        MultiLabelClfDataSet allTrainData = loadData(config,config.getString("input.trainData"));
        double[] instanceWeights = new double[allTrainData.getNumDataPoints()];
        Arrays.fill(instanceWeights,1.0);

        if (config.getBoolean("train.useInstanceWeights")){
            instanceWeights = loadInstanceWeights(config);
        }

        MultiLabelClfDataSet trainSetForEval = minibatch(allTrainData, instanceWeights, config.getInt("train.showProgress.sampleSize"),0+randomSeed).getFirst();

        MultiLabelClfDataSet validSet = loadData(config,config.getString("input.validData"));


        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(allTrainData);
        Serialization.serialize(support, Paths.get(output,"model_predictions",config.getString("output.modelFolder"),"models","support"));

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
            checkPoint = (CheckPoint) Serialization.deserialize(Paths.get(output,"model_predictions",config.getString("output.modelFolder"),"models","checkpoint"));
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
                Pair<MultiLabelClfDataSet, double[]> sampled = minibatch(allTrainData, instanceWeights, config.getInt("train.batchSize"),i+randomSeed);
                trainBatch = sampled.getFirst();
                IMLGBConfig imlgbConfig = new IMLGBConfig.Builder(trainBatch)
                        .learningRate(learningRate)
                        .minDataPerLeaf(minDataPerLeaf)
                        .numLeaves(numLeaves)
                        .numSplitIntervals(config.getInt("train.numSplitIntervals"))
                        .usePrior(config.getBoolean("train.usePrior"))
                        .numActiveFeatures(numActiveFeatures)
                        .build();

                trainer = new IMLGBTrainer(imlgbConfig, boosting, shouldStop);
                trainer.setInstanceWeights(sampled.getSecond());
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

            Serialization.serialize(checkPoint, Paths.get(output,"model_predictions",config.getString("output.modelFolder"),"models","checkpoint"));

            Serialization.serialize(boosting, Paths.get(output,"model_predictions",config.getString("output.modelFolder"),"models","classifier"));


            if (numLabelsLeftToTrain==0){
                logger.info("all label training finished");
                break;
            }
        }

        logger.info("training done");
        logger.info(stopWatch.toString());



        File analysisFolder = Paths.get(output, "model_predictions",config.getString("output.modelFolder"),"analysis").toFile();


        if (true){
            ObjectMapper objectMapper = new ObjectMapper();
            List<LabelModel> labelModels = IMLGBInspector.getAllRules(boosting);
            new File(analysisFolder,"decision_rules").mkdirs();

            for (int l=0;l<boosting.getNumClasses();l++){
                objectMapper.writeValue(Paths.get(analysisFolder.toString(), "decision_rules", l+".json").toFile(),labelModels.get(l));
            }

        }



        boolean topFeaturesToFile = true;

        if (topFeaturesToFile){
            logger.info("start writing top features");
            List<TopFeatures> topFeaturesList = IntStream.range(0,boosting.getNumClasses())
                    .mapToObj(k -> IMLGBInspector.topFeatures(boosting, k, Integer.MAX_VALUE)).collect(Collectors.toList());
            ObjectMapper mapper = new ObjectMapper();
            String file = "top_features.json";
            mapper.writeValue(new File(analysisFolder,file), topFeaturesList);

            StringBuilder sb = new StringBuilder();
            for (int l=0;l<boosting.getNumClasses();l++){
                sb.append("-------------------------").append("\n");
                sb.append(allTrainData.getLabelTranslator().toExtLabel(l)).append(":").append("\n");
                for (Feature feature: topFeaturesList.get(l).getTopFeatures()){
                    sb.append(feature.simpleString()).append(", ");
                }
                sb.append("\n");
            }
            FileUtils.writeStringToFile(new File(analysisFolder, "top_features.txt"), sb.toString());

            logger.info("finish writing top features");
        }

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



    private static Pair<MultiLabelClfDataSet,double[]> minibatch(MultiLabelClfDataSet allData, double[] instanceWeights, int minibatchSize, int interation){
        List<Integer> all = IntStream.range(0, allData.getNumDataPoints()).boxed().collect(Collectors.toList());
        Collections.shuffle(all, new Random(interation));
        List<Integer> keep = all.stream().limit(minibatchSize).collect(Collectors.toList());
        double[] subsetWeights = keep.stream().mapToDouble(i->instanceWeights[i]).toArray();
        return new Pair<>(DataSetUtil.sampleData(allData, keep),subsetWeights);
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
