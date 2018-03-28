package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.SafeDivide;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.*;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Calibration {
    public static void main(Config config, Logger logger) throws Exception{

//
//        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testSet"),DataSetType.ML_CLF_SPARSE,true);
//        IMLGradientBoosting boosting = (IMLGradientBoosting)Serialization.deserialize(config.getString("input.model"));
//
//        original(boosting, test, logger);
//
//        logger.info("start calibrating set probability");
//        IMLGBIsotonicScaling isotonicScaling = new IMLGBIsotonicScaling(boosting, test);
//        logger.info("finish calibrating set probability");
//
//
//        isoCalibration(boosting, test, isotonicScaling, logger);
//
//        labelUncalibration(boosting,test,logger);
//
//        labelIsoCalibration(boosting, test, logger, config);
//
//
//
//        Serialization.serialize(isotonicScaling, new File(config.getString("out"),"set_calibration"));
//



    }

//    private static void original( IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, Logger logger) throws Exception{
//
//        int numIntervals = 10;
//        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = new SubsetAccPredictor(boosting);
//
//
//        final  PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;
//        List<Result> results = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
//                .mapToObj(i->{
//                    Result result = new Result();
//                    Vector vector = dataSet.getRow(i);
//                    MultiLabel multiLabel = pluginPredictor.predict(vector);
//
//                    double probability = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
//                    result.probability = probability;
//                    result.correctness = multiLabel.equals(dataSet.getMultiLabels()[i]);
//                    return result;
//                }).collect(Collectors.toList());
//
//        double intervalSize = 1.0/numIntervals;
//        DecimalFormat decimalFormat = new DecimalFormat("#0.00");
//        StringBuilder stringBuilder = new StringBuilder();
//        stringBuilder.append("uncalibrated set probability\n");
//        stringBuilder.append("\ninterval"+"\t"+"total"+"\t"+"correct"+"\t\t"+"incorrect"+"\t"+"accuracy"+"\t"+"average confidence\n");
//        for (int i=0;i<numIntervals;i++){
//            double left = intervalSize*i;
//            double right = intervalSize*(i+1);
//            List<Result> matched = results.stream().filter(result -> (result.probability>=left && result.probability<right)).collect(Collectors.toList());
//            if (i==numIntervals-1){
//                matched = results.stream().filter(result -> (result.probability>=left && result.probability<=right)).collect(Collectors.toList());
//            }
//            int numPos = (int)matched.stream().filter(res->res.correctness).count();
//            int numNeg = matched.size()-numPos;
//            double aveProb = matched.stream().mapToDouble(res->res.probability).average().orElse(0);
//            double accuracy = SafeDivide.divide(numPos,matched.size(), 0);
//            String st = "["+decimalFormat.format(left)+", "+decimalFormat.format(right)+")"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb)+"\n";
//            if (i==numIntervals-1){
//                st = "["+decimalFormat.format(left)+", "+decimalFormat.format(right)+"]"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb)+"\n";
//            }
//            stringBuilder.append(st);
//        }
//        logger.info(stringBuilder.toString());
//
//    }
//
//
//    private static void isoCalibration(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, IMLGBIsotonicScaling scaling, Logger logger) throws Exception{
//        int numIntervals = 10;
//        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = new SubsetAccPredictor(boosting);
//        final  PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;
//        List<Result> results = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
//                .mapToObj(i->{
//                    Result result = new Result();
//                    Vector vector = dataSet.getRow(i);
//                    MultiLabel multiLabel = pluginPredictor.predict(vector);
//
//                    double probability = scaling.calibratedProb(dataSet.getRow(i),multiLabel);
//                    result.probability = probability;
//                    result.correctness = multiLabel.equals(dataSet.getMultiLabels()[i]);
//                    result.intId =i;
//                    return result;
//                }).collect(Collectors.toList());
//
//        double intervalSize = 1.0/numIntervals;
//        DecimalFormat decimalFormat = new DecimalFormat("#0.00");
//        StringBuilder stringBuilder = new StringBuilder();
//        stringBuilder.append("calibrated set probability\n");
//        stringBuilder.append("\ninterval"+"\t"+"total"+"\t"+"correct"+"\t\t"+"incorrect"+"\t"+"accuracy"+"\t"+"average confidence\n");
//        for (int i=0;i<numIntervals;i++){
//            double left = intervalSize*i;
//            double right = intervalSize*(i+1);
//            List<Result> matched = results.stream().filter(result -> (result.probability>=left && result.probability<right)).collect(Collectors.toList());
//            if (i==numIntervals-1){
//                matched = results.stream().filter(result -> (result.probability>=left && result.probability<=right)).collect(Collectors.toList());
//            }
//            int numPos = (int)matched.stream().filter(res->res.correctness).count();
//            int numNeg = matched.size()-numPos;
//            double aveProb = matched.stream().mapToDouble(res->res.probability).average().orElse(0);
//            double accuracy = SafeDivide.divide(numPos,matched.size(), 0);
//            String st = "["+decimalFormat.format(left)+", "+decimalFormat.format(right)+")"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb)+"\n";
//            if (i==numIntervals-1){
//                st = "["+decimalFormat.format(left)+", "+decimalFormat.format(right)+"]"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb)+"\n";
//            }
//            stringBuilder.append(st);
//        }
//        logger.info(stringBuilder.toString());
//
//
//    }
//
//    private static void labelIsoCalibration(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, Logger logger, Config config)throws Exception{
//        logger.info("start calibrating label probability ");
//        IMLGBLabelIsotonicScaling imlgbLabelIsotonicScaling = new IMLGBLabelIsotonicScaling(boosting, dataSet);
//        logger.info("finish calibrating label probability");
//        IMLGBLabelIsotonicScaling.BucketInfo total = imlgbLabelIsotonicScaling.individualProbs(dataSet);
//
//        double[] counts = total.getCounts();
//        double[] correct = total.getSums();
//        double[] sumProbs = total.getSumProbs();
//        double[] accs = new double[counts.length];
//        double[] average_confidence = new double[counts.length];
//
//        for (int i = 0; i < counts.length; i++) {
//            accs[i] = correct[i] / counts[i];
//        }
//        for (int j = 0; j < counts.length; j++) {
//            average_confidence[j] = sumProbs[j] / counts[j];
//        }
//
//        DecimalFormat decimalFormat = new DecimalFormat("#0.0000");
//        StringBuilder sb = new StringBuilder();
//        sb.append("calibrated label probabilities\n");
//        sb.append("interval\t\t").append("total\t\t").append("correct\t\t").append("incorrect\t\t").append("accuracy\t\t").append("average confidence\n");
//        for (int i = 0; i < 10; i++) {
//            sb.append("[").append(decimalFormat.format(i * 0.1)).append(",")
//                    .append(decimalFormat.format((i + 1) * 0.1)).append("]")
//                    .append("\t\t").append(counts[i]).append("\t\t").append(correct[i]).append("\t\t")
//                    .append(counts[i] - correct[i]).append("\t\t").append(decimalFormat.format(accs[i])).append("\t\t")
//                    .append(decimalFormat.format(average_confidence[i])).append("\n");
//
//        }
//        logger.info(sb.toString());
//        Serialization.serialize(imlgbLabelIsotonicScaling, new File(config.getString("out"),"label_calibration"));
//
//    }
//
//    private static void labelUncalibration(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, Logger logger)throws Exception{
//
//        final int numBuckets = 10;
//        double bucketLength = 1.0/numBuckets;
//
//        BucketInfo empty = new BucketInfo(numBuckets);
//        BucketInfo total;
//        total = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
//                .mapToObj(i->{
//                    double[] probs = boosting.predictClassProbs(dataSet.getRow(i));
//                    double[] count = new double[numBuckets];
//                    double[] sum = new double[numBuckets];
//                    double[] sumProbs = new double[numBuckets];
//                    for (int a=0;a<probs.length;a++){
//                        int index = (int)Math.floor(probs[a]/bucketLength);
//                        if (index<0){
//                            index=0;
//                        }
//                        if (index>=numBuckets){
//                            index = numBuckets-1;
//                        }
//                        count[index] += 1;
//                        sumProbs[index] += probs[a];
//                        if (dataSet.getMultiLabels()[i].matchClass(a)){
//                            sum[index] += 1;
//                        } else {
//                            sum[index] += 0;
//                        }
//                    }
//                    return new BucketInfo(count, sum, sumProbs);
//                }).reduce(empty, BucketInfo::add, BucketInfo::add);
//
//        double[] counts = total.getCounts();
//        double[] correct = total.getSums();
//        double[] sumProbs = total.getSumProbs();
//        double[] accs = new double[counts.length];
//        double[] average_confidence = new double[counts.length];
//
//        for (int i = 0; i < counts.length; i++) {
//            accs[i] = correct[i] / counts[i];
//        }
//        for (int j = 0; j < counts.length; j++) {
//            average_confidence[j] = sumProbs[j] / counts[j];
//        }
//
//        DecimalFormat decimalFormat = new DecimalFormat("#0.0000");
//        StringBuilder sb = new StringBuilder();
//        sb.append("uncalibrated label probabilities\n");
//        sb.append("interval\t\t").append("total\t\t").append("correct\t\t").append("incorrect\t\t").append("accuracy\t\t").append("average confidence\n");
//        for (int i = 0; i < 10; i++) {
//            sb.append("[").append(decimalFormat.format(i * 0.1)).append(",")
//                    .append(decimalFormat.format((i + 1) * 0.1)).append("]")
//                    .append("\t\t").append(counts[i]).append("\t\t").append(correct[i]).append("\t\t")
//                    .append(counts[i] - correct[i]).append("\t\t").append(decimalFormat.format(accs[i])).append("\t\t")
//                    .append(decimalFormat.format(average_confidence[i])).append("\n");
//
//        }
//        logger.info(sb.toString());
//
//
//    }
//
//
//
//
//
//
//
//
//
//    static class Result{
//        int intId;
//        double probability;
//        boolean correctness;
//    }
}
