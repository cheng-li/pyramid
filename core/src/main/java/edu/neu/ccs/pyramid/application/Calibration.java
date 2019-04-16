package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.*;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.CalibrationEval;

import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.optimization.EarlyStopper;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostOptimizer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.util.*;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;


import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * General calibration app for multi-label classifiers that output set scores
 */
public class Calibration {
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

        if (config.getBoolean("calibrate")){
            calibrate(config, logger);
        }

        if (config.getBoolean("test")){
            test(config, logger);
        }

        if (fileHandler!=null){
            fileHandler.close();
        }
    }
    public static void main(String[] args) throws Exception {
        Config config = new Config(args[0]);
        main(config);

    }


    private static void calibrate(Config config, Logger logger) throws Exception{

        logger.info("start training calibrator");

        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"), DataSetType.ML_CLF_SEQ_SPARSE, true);

        MultiLabelClfDataSet cal = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.calibData"), DataSetType.ML_CLF_SEQ_SPARSE, true);

        MultiLabel[] calPred = loadPredictions(config.getString("input.calibPredictions"));

        int[] calIds = loadIds(config.getString("input.calibPredictions"));
        Vector[] calibScore = loadFeatures(config.getString("input.calibPredictions"));

        Pair<RegDataSet,PredictionFeatureExtractor> pair = createCaliData(cal,calPred,calibScore, calIds, train);
        RegDataSet calibRegData = pair.getFirst();
        PredictionFeatureExtractor predictionFeatureExtractor = pair.getSecond();
        int[] monotonicity = pair.getSecond().featureMonotonicity();

        MultiLabelClfDataSet valid = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.validData"), DataSetType.ML_CLF_SEQ_SPARSE, true);

        MultiLabel[] validPred = loadPredictions(config.getString("input.validPredictions"));

        int[] validIds = loadIds(config.getString("input.validPredictions"));
        Vector[] validScore = loadFeatures(config.getString("input.validPredictions"));

        RegDataSet validRegData = createCaliData(valid,validPred,validScore, validIds, predictionFeatureExtractor).getFirst();

        LSBoost lsBoost = trainCalibrator(calibRegData,validRegData,monotonicity);

        VectorCalibrator vectorCalibrator = new RegressorCalibrator(lsBoost);

        Serialization.serialize(vectorCalibrator,Paths.get(config.getString("output.dir"),"set_calibrator").toFile());
        Serialization.serialize(predictionFeatureExtractor,Paths.get(config.getString("output.dir"),"prediction_feature_extractor").toFile());

        logger.info("finish training calibrator");
    }

    private static LSBoost trainCalibrator(RegDataSet calib, RegDataSet valid, int[] monotonicity){
        LSBoost lsBoost = new LSBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(10).setMinDataPerLeaf(5);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSBoostOptimizer optimizer = new LSBoostOptimizer(lsBoost, calib, regTreeFactory, calib.getLabels());
        if (true){
            int[][] mono = new int[1][calib.getNumFeatures()];
            mono[0] = monotonicity;
            optimizer.setMonotonicity(mono);
        }
        optimizer.setShrinkage(0.1);
        optimizer.initialize();
        EarlyStopper earlyStopper = new EarlyStopper(EarlyStopper.Goal.MINIMIZE,5);
        LSBoost bestModel = null;
        for (int i = 1; i<1000; i++){
            optimizer.iterate();

            if (i%10==0){
                double mse = MSE.mse(lsBoost, valid);
                earlyStopper.add(i,mse);
                if (earlyStopper.getBestIteration()==i){
                    try {
                        bestModel = (LSBoost) Serialization.deepCopy(lsBoost);
                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch (ClassNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                if (earlyStopper.shouldStop()){
                    break;
                }
            }
        }
        return bestModel;
    }


    private static MultiLabel[] loadPredictions(String file) throws Exception{
        List<String> lines = FileUtils.readLines(new File(file));
        MultiLabel[] predictions = new MultiLabel[lines.size()];
        for (int i=0;i<lines.size();i++){
            predictions[i] = new MultiLabel();
            String[] split = lines.get(i).split(Pattern.quote("("))[0].split(Pattern.quote(":"))[1].replace(" ","").split(",");
            for (String l: split){
                if (!l.isEmpty()){
                    predictions[i].addLabel(Integer.parseInt(l));
                }

            }
        }
        return predictions;
    }


    private static Vector[] loadFeatures(String file) throws Exception{
        List<String> lines = FileUtils.readLines(new File(file));
        Vector[] scores = new Vector[lines.size()];
        for (int i=0;i<lines.size();i++){
            String split = lines.get(i).split(Pattern.quote("("))[1].replace(")","");
            String[] features = split.split(",");
            Vector vector = new DenseVector(features.length);
            for (int j=0;j<features.length;j++){
                vector.set(j,Double.parseDouble(features[j].trim()));
            }
            scores[i] = vector;
        }
        return scores;
    }

    private static int[] loadIds(String file) throws Exception{
        List<String> lines = FileUtils.readLines(new File(file));
        int[] ids = new int[lines.size()];
        for (int i=0;i<lines.size();i++){
            String split = lines.get(i).split(Pattern.quote(":"))[0];
            ids[i] = Integer.parseInt(split);
        }
        return ids;
    }

    private static Pair<RegDataSet,PredictionFeatureExtractor> createCaliData(MultiLabelClfDataSet multiLabelClfDataSet, MultiLabel[] predictions, Vector[] scores, int[] ids, MultiLabelClfDataSet train){

        int numFeatures = 2 + train.getNumClasses() + scores[0].size();

        RegDataSet regDataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(predictions.length)
                .numFeatures(numFeatures)
                .build();

        List<PredictionFeatureExtractor> extractors = new ArrayList<>();
        extractors.add(new PriorFeatureExtractor(train));
        extractors.add(new CardFeatureExtractor());
        extractors.add(new LabelBinaryFeatureExtractor(train.getNumClasses(),train.getLabelTranslator()));
        List<Integer> featureIds = IntStream.range(0,scores[0].size()).boxed().collect(Collectors.toList());
        extractors.add(new InstanceFeatureExtractor(featureIds,train.getFeatureList()));
        PredictionFeatureExtractor predictionFeatureExtractor = new CombinedPredictionFeatureExtractor(extractors);

        for (int i=0;i<predictions.length;i++){
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.multiLabel = predictions[i];
            predictionCandidate.x = scores[i];
            Vector row = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            for (int j=0;j<regDataSet.getNumFeatures();j++){
                regDataSet.setFeatureValue(i,j,row.get(j));
            }
            int id = ids[i];
            if (multiLabelClfDataSet.getMultiLabels()[id].equals(predictions[i])){
                regDataSet.setLabel(i,1);
            } else {
                regDataSet.setLabel(i,0);
            }
        }
        return new Pair<>(regDataSet,predictionFeatureExtractor);
    }



    private static Pair<RegDataSet,PredictionFeatureExtractor> createCaliData(MultiLabelClfDataSet multiLabelClfDataSet, MultiLabel[] predictions, Vector[] scores, int[] ids, PredictionFeatureExtractor predictionFeatureExtractor){

        int numFeatures = predictionFeatureExtractor.featureMonotonicity().length;

        RegDataSet regDataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(predictions.length)
                .numFeatures(numFeatures)
                .build();

        for (int i=0;i<predictions.length;i++){
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.multiLabel = predictions[i];
            predictionCandidate.x = scores[i];
            Vector row = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            for (int j=0;j<regDataSet.getNumFeatures();j++){
                regDataSet.setFeatureValue(i,j,row.get(j));
            }
            int id = ids[i];
            if (multiLabelClfDataSet.getMultiLabels()[id].equals(predictions[i])){
                regDataSet.setLabel(i,1);
            } else {
                regDataSet.setLabel(i,0);
            }
        }
        return new Pair<>(regDataSet,predictionFeatureExtractor);
    }


    private static void test(Config config, Logger logger) throws Exception{
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"), DataSetType.ML_CLF_SEQ_SPARSE, true);

        VectorCalibrator setCalibrator = (VectorCalibrator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"set_calibrator").toFile());

        PredictionFeatureExtractor predictionFeatureExtractor = (PredictionFeatureExtractor) Serialization.deserialize(Paths.get(config.getString("output.dir"),"prediction_feature_extractor").toFile());

        MultiLabel[] predictions = loadPredictions(config.getString("input.testPredictions"));
        Vector[] scores = loadFeatures(config.getString("input.testPredictions"));

        int[] ids = loadIds(config.getString("input.testPredictions"));

        RegDataSet testRegData = createCaliData(test,predictions,scores, ids, predictionFeatureExtractor).getFirst();
        if (true) {
            logger.info("calibration performance on test set");

            List<CalibrationDataGenerator.CalibrationInstance> instances = IntStream.range(0, test.getNumDataPoints()).parallel()
                    .boxed().map(i -> {

                        CalibrationDataGenerator.CalibrationInstance instance = new CalibrationDataGenerator.CalibrationInstance();
                        instance.vector = testRegData.getRow(i);
                        instance.correctness=testRegData.getLabels()[i];
                        return instance;
                    })
                    .collect(Collectors.toList());
            eval(instances, setCalibrator, logger);
        }



        double[] calibratedProbs = IntStream.range(0,testRegData.getNumDataPoints()).parallel()
                .mapToDouble(i->setCalibrator.calibrate(testRegData.getRow(i))).toArray();
        String result = PrintUtil.toMutipleLines(calibratedProbs);
        FileUtils.writeStringToFile(Paths.get(config.getString("output.dir"),"test_calibrated_probabilities.txt").toFile(),result);
        logger.info("calibrated probabilities for test predictions are saved in "+Paths.get(config.getString("output.dir"),"test_calibrated_probabilities.txt").toFile().getAbsolutePath());

        Map<Integer,Res> bestPredictions = new HashMap<>();
        for (int i=0;i<ids.length;i++){
            int id = ids[i];
            MultiLabel prediction = predictions[i];
            double confidence = calibratedProbs[i];
            Res res = new Res(prediction,confidence);
            if (bestPredictions.containsKey(id)){
                Res old = bestPredictions.get(id);
                if (res.confidence>old.confidence){
                    bestPredictions.put(id,res);
                }
            } else {
                bestPredictions.put(id,res);
            }
        }

        MultiLabel[] rerankedPredictions = new MultiLabel[test.getNumDataPoints()];
        for (int i=0;i<rerankedPredictions.length;i++){
            rerankedPredictions[i] = bestPredictions.get(i).prediction;
        }

        StringBuilder stringBuilder = new StringBuilder();
        for (int i=0;i<rerankedPredictions.length;i++){
            stringBuilder.append(i).append(": ").append(rerankedPredictions[i].toSimpleString()).append("\n");
        }
        FileUtils.writeStringToFile(Paths.get(config.getString("output.dir"),"test_reranked_predictions.txt").toFile(),stringBuilder.toString());

        logger.info("test performance");
        logger.info(new MLMeasures(test.getNumClasses(),test.getMultiLabels(),rerankedPredictions).toString());
    }


    private static BRCalibration.CaliRes eval(List<CalibrationDataGenerator.CalibrationInstance> predictions, VectorCalibrator calibrator, Logger logger){
        double mse = CalibrationEval.mse(generateStream(predictions,calibrator));
        double ace = CalibrationEval.absoluteError(generateStream(predictions,calibrator),10);
        double sharpness = CalibrationEval.sharpness(generateStream(predictions,calibrator),10);
        logger.info("mse="+mse);
        logger.info("absolute calibration error="+ace);
        logger.info("square calibration error="+CalibrationEval.squareError(generateStream(predictions,calibrator),10));
        logger.info("sharpness="+sharpness);
        logger.info("uncertainty="+CalibrationEval.variance(generateStream(predictions,calibrator)));
        logger.info(Displayer.displayCalibrationResult(generateStream(predictions,calibrator)));
        BRCalibration.CaliRes caliRes = new BRCalibration.CaliRes();
        caliRes.mse = mse;
        caliRes.ace= ace;
        caliRes.sharpness = sharpness;
        return caliRes;
    }



    private static Stream<Pair<Double,Integer>> generateStream(List<CalibrationDataGenerator.CalibrationInstance> predictions, VectorCalibrator vectorCalibrator){
        return predictions.stream()
                .parallel().map(pred->new Pair<>(vectorCalibrator.calibrate(pred.vector),(int)pred.correctness));
    }


    private static class Res{
        public Res(MultiLabel prediction, double confidence) {
            this.prediction = prediction;
            this.confidence = confidence;
        }

        MultiLabel prediction;
        double confidence;
    }

}
