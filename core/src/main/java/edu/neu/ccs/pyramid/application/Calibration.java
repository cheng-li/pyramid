package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.SafeDivide;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.*;
import edu.neu.ccs.pyramid.multilabel_classification.thresholding.TunedMarginalClassifier;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Calibration {
    public static void main(Config config, Logger logger) throws Exception{


        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testSet"),DataSetType.ML_CLF_SPARSE,true);
        IMLGradientBoosting boosting = (IMLGradientBoosting)Serialization.deserialize(config.getString("input.model"));
        IMLGBIsotonicScaling isotonicScaling = new IMLGBIsotonicScaling(boosting, test);

        logger.info("====original probabilities on the test set====");

        original(boosting, test, logger);


        logger.info("====Isotonic regression calibrated probabilities on the test set====");

        isoCalibration(boosting, test, isotonicScaling, logger);

        Serialization.serialize(isotonicScaling, new File(config.getString("out"),"calibration"));


    }

    private static void original( IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, Logger logger) throws Exception{

        int numIntervals = 10;
        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = new SubsetAccPredictor(boosting);


        final  PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;
        List<Result> results = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    Result result = new Result();
                    Vector vector = dataSet.getRow(i);
                    MultiLabel multiLabel = pluginPredictor.predict(vector);

                    double probability = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
                    result.probability = probability;
                    result.correctness = multiLabel.equals(dataSet.getMultiLabels()[i]);
                    return result;
                }).collect(Collectors.toList());

        double intervalSize = 1.0/numIntervals;
        DecimalFormat decimalFormat = new DecimalFormat("#0.00");
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("\ninterval"+"\t"+"total"+"\t"+"correct"+"\t\t"+"incorrect"+"\t"+"accuracy"+"\t"+"average confidence\n");
        for (int i=0;i<numIntervals;i++){
            double left = intervalSize*i;
            double right = intervalSize*(i+1);
            List<Result> matched = results.stream().filter(result -> (result.probability>=left && result.probability<right)).collect(Collectors.toList());
            if (i==numIntervals-1){
                matched = results.stream().filter(result -> (result.probability>=left && result.probability<=right)).collect(Collectors.toList());
            }
            int numPos = (int)matched.stream().filter(res->res.correctness).count();
            int numNeg = matched.size()-numPos;
            double aveProb = matched.stream().mapToDouble(res->res.probability).average().orElse(0);
            double accuracy = SafeDivide.divide(numPos,matched.size(), 0);
            String st = "["+decimalFormat.format(left)+", "+decimalFormat.format(right)+")"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb)+"\n";
            if (i==numIntervals-1){
                st = "["+decimalFormat.format(left)+", "+decimalFormat.format(right)+"]"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb)+"\n";
            }
            stringBuilder.append(st);
        }
        logger.info(stringBuilder.toString());

    }


    private static void isoCalibration(IMLGradientBoosting boosting, MultiLabelClfDataSet dataSet, IMLGBIsotonicScaling scaling, Logger logger) throws Exception{
        int numIntervals = 10;
        PluginPredictor<IMLGradientBoosting> pluginPredictorTmp = new SubsetAccPredictor(boosting);
        final  PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;
        List<Result> results = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    Result result = new Result();
                    Vector vector = dataSet.getRow(i);
                    MultiLabel multiLabel = pluginPredictor.predict(vector);

                    double probability = scaling.calibratedProb(dataSet.getRow(i),multiLabel);
                    result.probability = probability;
                    result.correctness = multiLabel.equals(dataSet.getMultiLabels()[i]);
                    result.intId =i;
                    return result;
                }).collect(Collectors.toList());

        double intervalSize = 1.0/numIntervals;
        DecimalFormat decimalFormat = new DecimalFormat("#0.00");
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("\ninterval"+"\t"+"total"+"\t"+"correct"+"\t\t"+"incorrect"+"\t"+"accuracy"+"\t"+"average confidence\n");
        for (int i=0;i<numIntervals;i++){
            double left = intervalSize*i;
            double right = intervalSize*(i+1);
            List<Result> matched = results.stream().filter(result -> (result.probability>=left && result.probability<right)).collect(Collectors.toList());
            if (i==numIntervals-1){
                matched = results.stream().filter(result -> (result.probability>=left && result.probability<=right)).collect(Collectors.toList());
            }
            int numPos = (int)matched.stream().filter(res->res.correctness).count();
            int numNeg = matched.size()-numPos;
            double aveProb = matched.stream().mapToDouble(res->res.probability).average().orElse(0);
            double accuracy = SafeDivide.divide(numPos,matched.size(), 0);
            String st = "["+decimalFormat.format(left)+", "+decimalFormat.format(right)+")"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb)+"\n";
            if (i==numIntervals-1){
                st = "["+decimalFormat.format(left)+", "+decimalFormat.format(right)+"]"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb)+"\n";
            }
            stringBuilder.append(st);
        }
        logger.info(stringBuilder.toString());


    }



    static class Result{
        int intId;
        double probability;
        boolean correctness;
    }
}
