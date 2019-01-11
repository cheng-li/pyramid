package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.Displayer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.*;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class BRGBCalibration {
    public static void main(Config config, Logger logger) throws Exception{


        MultiLabelClfDataSet valid = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.validSet"),DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet cali = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.calibrationSet"),DataSetType.ML_CLF_SPARSE,true);
        IMLGradientBoosting boosting = (IMLGradientBoosting)Serialization.deserialize(config.getString("input.model"));

        //original(boosting, test, logger);


        logger.info("start cardinality based set probability calibration");
        CardinalityCalibrator cardinalityCalibrator = new CardinalityCalibrator(boosting, cali);
        logger.info("finish cardinality based set probability calibration");

        logger.info("calibration performance on calibration set");
        displayCardinalityCalibration(boosting, cali, cardinalityCalibrator, logger);
        logger.info("calibration performance on validation set");
        displayCardinalityCalibration(boosting, valid, cardinalityCalibrator, logger);

        //labelUncalibration(boosting,test,logger);


        labelIsoCalibration(boosting, cali, logger, config);

        Serialization.serialize(cardinalityCalibrator, new File(config.getString("out"),"set_calibration"));


    }

    public static void original( IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, Logger logger) throws Exception{


        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = new SubsetAccPredictor(boosting);
        final  PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;

        Stream<Pair<Double,Integer>> stream = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    Pair<Double,Integer> pairOverall = new Pair<>();
                    Vector vector = dataSet.getRow(i);
                    MultiLabel multiLabel = pluginPredictor.predict(vector);
                    double prob = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
                    pairOverall.setFirst(prob);
                    pairOverall.setSecond(0);
                    if (multiLabel.equals(dataSet.getMultiLabels()[i])) {
                        pairOverall.setSecond(1);
                    }
                    return pairOverall;
                });

        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("uncalibrated set probability\n");

        stringBuilder.append(Displayer.displayCalibrationResult(stream));

        logger.info(stringBuilder.toString());

    }


    private static void displaySetCalibration(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, IMLGBIsotonicScaling scaling, Logger logger) throws Exception{
        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = new SubsetAccPredictor(boosting);
        final  PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;

        Stream<Pair<Double,Integer>> stream = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    Pair<Double,Integer> pairOverall = new Pair<>();
                    Vector vector = dataSet.getRow(i);
                    MultiLabel multiLabel = pluginPredictor.predict(vector);
                    double prob = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
                    double calibrated = scaling.calibratedProb(prob);
                    pairOverall.setFirst(calibrated);
                    pairOverall.setSecond(0);
                    if (multiLabel.equals(dataSet.getMultiLabels()[i])) {
                        pairOverall.setSecond(1);
                    }
                    return pairOverall;
                });


        logger.info(Displayer.displayCalibrationResult(stream));
    }


    public static void displayCardinalityCalibration(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, CardinalityCalibrator scaling, Logger logger) throws Exception{
        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = new SubsetAccPredictor(boosting);
        final  PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;

        Stream<Pair<Double,Integer>> stream = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    Pair<Double,Integer> pairOverall = new Pair<>();
                    Vector vector = dataSet.getRow(i);
                    MultiLabel multiLabel = pluginPredictor.predict(vector);
                    double prob = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
                    double calibratedProb = scaling.calibrate(prob, multiLabel.getNumMatchedLabels());
                    pairOverall.setFirst(calibratedProb);
                    pairOverall.setSecond(0);
                    if (multiLabel.equals(dataSet.getMultiLabels()[i])) {
                        pairOverall.setSecond(1);
                    }
                    return pairOverall;
                });


        logger.info(Displayer.displayCalibrationResult(stream));


    }


    private static void labelIsoCalibration(IMLGradientBoosting boosting,  MultiLabelClfDataSet caliSet, Logger logger, Config config)throws Exception{
        logger.info("start calibrating label probability ");
        IMLGBLabelIsotonicScaling imlgbLabelIsotonicScaling = new IMLGBLabelIsotonicScaling(boosting, caliSet);
        logger.info("finish calibrating label probability");

//        Stream<Pair<Double,Integer>> stream = IntStream.range(0, testSet.getNumDataPoints()).parallel()
//                .boxed().flatMap(i-> {
//                    double[] probs = boosting.predictClassProbs(testSet.getRow(i));
//                    double[] calibrated = imlgbLabelIsotonicScaling.calibratedClassProbs(probs);
//                    Stream<Pair<Double,Integer>> pairs = IntStream.range(0, probs.length).mapToObj(a -> {
//                        Pair<Double, Integer> pair = new Pair<>();
//                        pair.setFirst(calibrated[a]);
//                        pair.setSecond(0);
//                        if (testSet.getMultiLabels()[i].matchClass(a)) {
//                            pair.setSecond(1);
//                        }
//                        return pair;
//                    });
//                    return pairs;
//                });

        //StringBuilder sb = new StringBuilder();
        //sb.append("calibrated label probabilities\n");
        //sb.append(Displayer.displayCalibrationResult(stream));
        //logger.info(sb.toString());
        Serialization.serialize(imlgbLabelIsotonicScaling, new File(config.getString("out"),"label_calibration"));

    }


    private static void labelUncalibration(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, Logger logger)throws Exception{

        Stream<Pair<Double,Integer>> stream = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .boxed().flatMap(i-> {
                        double[] probs = boosting.predictClassProbs(dataSet.getRow(i));
                        Stream<Pair<Double,Integer>> pairs = IntStream.range(0, probs.length).mapToObj(a -> {
                            Pair<Double, Integer> pair = new Pair<>();
                            pair.setFirst(probs[a]);
                            pair.setSecond(0);
                            if (dataSet.getMultiLabels()[i].matchClass(a)) {
                                pair.setSecond(1);
                            }
                            return pair;
                        });
                        return pairs;
                });


        StringBuilder sb = new StringBuilder();
        sb.append("uncalibrated label probabilities\n");
        sb.append(Displayer.displayCalibrationResult(stream));
        logger.info(sb.toString());
    }


}
