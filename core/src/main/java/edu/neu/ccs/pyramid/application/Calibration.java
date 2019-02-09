package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.*;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.CalibrationEval;

import edu.neu.ccs.pyramid.util.*;
import org.apache.commons.io.FileUtils;


import java.io.File;
import java.io.Serializable;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
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

        MultiLabelClfDataSet cal = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.validData"), DataSetType.ML_CLF_SEQ_SPARSE, true);

        MultiLabel[] validPred = loadPredictions(config.getString("input.validPredictions"));

        int[] ids = loadIds(config.getString("input.validPredictions"));
        double[] validScore = loadScores(config.getString("input.validPredictions"));

        RegDataSet calibratorTrainData = createCaliData(cal,validPred,validScore, ids);

        VectorCalibrator setCalibrator = new VectorCardIsoSetCalibrator(calibratorTrainData, 0, 1);

        Serialization.serialize(setCalibrator,Paths.get(config.getString("output.dir"),"set_calibrator").toFile());

        logger.info("finish training calibrator");
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


    private static double[] loadScores(String file) throws Exception{
        List<String> lines = FileUtils.readLines(new File(file));
        double[] scores = new double[lines.size()];
        for (int i=0;i<lines.size();i++){
            String split = lines.get(i).split(Pattern.quote("("))[1].replace(")","");
            scores[i] = Double.parseDouble(split);
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

    private static RegDataSet createCaliData(MultiLabelClfDataSet multiLabelClfDataSet, MultiLabel[] predictions, double[] scores, int[] ids){
        RegDataSet regDataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(predictions.length)
                .numFeatures(2)
                .build();
        for (int i=0;i<predictions.length;i++){
            regDataSet.setFeatureValue(i,0,scores[i]);
            regDataSet.setFeatureValue(i,1,predictions[i].getNumMatchedLabels());
            int id = ids[i];
            if (multiLabelClfDataSet.getMultiLabels()[id].equals(predictions[i])){
                regDataSet.setLabel(i,1);
            } else {
                regDataSet.setLabel(i,0);
            }
        }
        return regDataSet;
    }


    private static void test(Config config, Logger logger) throws Exception{
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"), DataSetType.ML_CLF_SEQ_SPARSE, true);

        VectorCalibrator setCalibrator = (VectorCalibrator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"set_calibrator").toFile());

        MultiLabel[] predictions = loadPredictions(config.getString("input.testPredictions"));
        double[] scores = loadScores(config.getString("input.testPredictions"));

        int[] ids = loadIds(config.getString("input.testPredictions"));

        RegDataSet calTestData = createCaliData(test,predictions,scores, ids);
        if (true) {
            logger.info("calibration performance on test set");

            List<CalibrationDataGenerator.CalibrationInstance> instances = IntStream.range(0, test.getNumDataPoints()).parallel()
                    .boxed().map(i -> {

                        CalibrationDataGenerator.CalibrationInstance instance = new CalibrationDataGenerator.CalibrationInstance();
                        instance.vector = calTestData.getRow(i);
                        instance.correctness=calTestData.getLabels()[i];
                        return instance;
                    })
                    .collect(Collectors.toList());
            eval(instances, setCalibrator, logger);
        }



        double[] calibratedProbs = IntStream.range(0,test.getNumDataPoints()).parallel()
                .mapToDouble(i->setCalibrator.calibrate(calTestData.getRow(i))).toArray();
        String result = PrintUtil.toMutipleLines(calibratedProbs);
        FileUtils.writeStringToFile(Paths.get(config.getString("output.dir"),"test_calibrated_probabilities.txt").toFile(),result);
        logger.info("calibrated probabilities for test predictions are saved in "+Paths.get(config.getString("output.dir"),"test_calibrated_probabilities.txt").toFile().getAbsolutePath());
    }


    private static BRLRCalibration.CaliRes eval(List<CalibrationDataGenerator.CalibrationInstance> predictions, VectorCalibrator calibrator, Logger logger){
        double mse = CalibrationEval.mse(generateStream(predictions,calibrator));
        double ace = CalibrationEval.absoluteError(generateStream(predictions,calibrator),10);
        double sharpness = CalibrationEval.sharpness(generateStream(predictions,calibrator),10);
        logger.info("mse="+mse);
        logger.info("absolute calibration error="+ace);
        logger.info("square calibration error="+CalibrationEval.squareError(generateStream(predictions,calibrator),10));
        logger.info("sharpness="+sharpness);
        logger.info("variance="+CalibrationEval.variance(generateStream(predictions,calibrator)));
        logger.info(Displayer.displayCalibrationResult(generateStream(predictions,calibrator)));
        BRLRCalibration.CaliRes caliRes = new BRLRCalibration.CaliRes();
        caliRes.mse = mse;
        caliRes.ace= ace;
        caliRes.sharpness = sharpness;
        return caliRes;
    }



    private static Stream<Pair<Double,Integer>> generateStream(List<CalibrationDataGenerator.CalibrationInstance> predictions, VectorCalibrator vectorCalibrator){
        return predictions.stream()
                .parallel().map(pred->new Pair<>(vectorCalibrator.calibrate(pred.vector),(int)pred.correctness));
    }

}
