package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.SafeDivide;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.*;
import edu.neu.ccs.pyramid.multilabel_classification.thresholding.TunedMarginalClassifier;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.text.DecimalFormat;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Calibration {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println("=========original probabilities==============");
        original(config);
        calibration(config);

    }

    private static void original(Config config) throws Exception{
        IMLGradientBoosting boosting = (IMLGradientBoosting)Serialization.deserialize(config.getString("input.model"));
        File modelFolder = (new File(config.getString("input.model"))).getParentFile();
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.data"), DataSetType.ML_CLF_SPARSE,
                true);
        int numIntervals = 10;
        String predictTarget = config.getString("predict.Target");
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
                TunedMarginalClassifier tunedMarginalClassifier = (TunedMarginalClassifier)Serialization.deserialize(new File(modelFolder, "predictor_macro_f"));
                pluginPredictorTmp = new MacroF1Predictor(boosting,tunedMarginalClassifier);
                break;
            default:
                throw new IllegalArgumentException("unknown prediction target measure "+predictTarget);
        }

        final  PluginPredictor<IMLGradientBoosting> pluginPredictor = pluginPredictorTmp;
        List<Result> results = IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToObj(i->{
                    Result result = new Result();
                    Vector vector = dataSet.getRow(i);
                    MultiLabel multiLabel = pluginPredictor.predict(vector);

                    double probability;
                    switch (predictTarget){
                        case "subsetAccuracy":
                            probability = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
                            break;
                        case "hammingLoss":
                            probability = boosting.predictAssignmentProbWithoutConstraint(vector, multiLabel);
                            break;
                        case "instanceFMeasure":
                            probability = boosting.predictAssignmentProbWithConstraint(vector, multiLabel);
                            break;
                        case "macroFMeasure":
                            probability = boosting.predictAssignmentProbWithoutConstraint(vector, multiLabel);
                            break;
                        default:
                            throw new IllegalArgumentException("unknown prediction target measure "+predictTarget);
                    }
                    result.probability = probability;
                    result.correctness = multiLabel.equals(dataSet.getMultiLabels()[i]);
                    return result;
                }).collect(Collectors.toList());

        double intervalSize = 1.0/numIntervals;
        DecimalFormat decimalFormat = new DecimalFormat("#0.00");
        System.out.println("interval"+"\t"+"total"+"\t"+"correct"+"\t\t"+"incorrect"+"\t"+"accuracy"+"\t"+"average confidence");
        for (int i=0;i<numIntervals;i++){
            double left = intervalSize*i;
            double right = intervalSize*(i+1);
            List<Result> matched = results.stream().filter(result -> (result.probability>=left && result.probability<=right)).collect(Collectors.toList());
            int numPos = (int)matched.stream().filter(res->res.correctness).count();
            int numNeg = matched.size()-numPos;
            double aveProb = matched.stream().mapToDouble(res->res.probability).average().orElse(0);
            double accuracy = SafeDivide.divide(numPos,matched.size(), 0);
            System.out.println("["+decimalFormat.format(left)+", "+decimalFormat.format(right)+"]"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb));
        }

    }

    private static void calibration(Config config) throws Exception{
        IMLGradientBoosting boosting = (IMLGradientBoosting)Serialization.deserialize(config.getString("input.model"));
        File modelFolder = (new File(config.getString("input.model"))).getParentFile();
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.data"), DataSetType.ML_CLF_SPARSE,
                true);

        IMLGBScaling scaling = new IMLGBScaling(boosting, dataSet);

        int numIntervals = 10;
        String predictTarget = config.getString("predict.Target");
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
                TunedMarginalClassifier tunedMarginalClassifier = (TunedMarginalClassifier)Serialization.deserialize(new File(modelFolder, "predictor_macro_f"));
                pluginPredictorTmp = new MacroF1Predictor(boosting,tunedMarginalClassifier);
                break;
            default:
                throw new IllegalArgumentException("unknown prediction target measure "+predictTarget);
        }

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
        System.out.println("interval"+"\t"+"total"+"\t"+"correct"+"\t\t"+"incorrect"+"\t"+"accuracy"+"\t"+"average confidence");
        for (int i=0;i<numIntervals;i++){
            double left = intervalSize*i;
            double right = intervalSize*(i+1);
            List<Result> matched = results.stream().filter(result -> (result.probability>=left && result.probability<=right)).collect(Collectors.toList());
            int numPos = (int)matched.stream().filter(res->res.correctness).count();
            int numNeg = matched.size()-numPos;
            double aveProb = matched.stream().mapToDouble(res->res.probability).average().orElse(0);
            double accuracy = SafeDivide.divide(numPos,matched.size(), 0);
            System.out.println("["+decimalFormat.format(left)+", "+decimalFormat.format(right)+"]"+"\t"+matched.size()+"\t"+numPos+"\t\t"+numNeg+"\t\t"+decimalFormat.format(accuracy)+"\t\t"+decimalFormat.format(aveProb));
        }

        dump(0.7,0.8, dataSet, config, results);
        dump(0.8,0.9, dataSet, config, results);
        dump(0.9,1, dataSet, config, results);
        
    }

    private static void dump(double left, double right, MultiLabelClfDataSet dataSet, Config config, List<Result> results){
        File data = new File(config.getString("input.data")+"_calibration_"+left+"-"+right);
        List<Integer>ids = results.stream().filter(res->res.probability>=left&&res.probability<=right).map(res->res.intId).collect(Collectors.toList());
        MultiLabelClfDataSet subset = DataSetUtil.sampleData(dataSet,ids);
        TRECFormat.save(subset, data);
        System.out.println("saved data with calibrated probabilities between "+left+" and "+right+" to "+data.getAbsolutePath());
    }

    static class Result{
        int intId;
        double probability;
        boolean correctness;
    }
}
