package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.linear_regression.ElasticNetLinearRegOptimizer;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.PrintUtil;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Linear Regression with Elasticnet (L1+L2) regularization
 * Created by chengli on 2/25/16.
 */
public class LinearRegElasticNet {

    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        String output = config.getString("output.folder");
        new File(output).mkdirs();

        String sparsity = config.getString("featureMatrix.sparsity").toLowerCase();
        DataSetType dataSetType = null;
        switch (sparsity){
            case "dense":
                dataSetType = DataSetType.REG_DENSE;
                break;
            case "sparse":
                dataSetType = DataSetType.REG_SPARSE;
                break;
            default:
                throw new IllegalArgumentException("featureMatrix.sparsity can be either dense or sparse");
        }

        RegDataSet trainSet = TRECFormat.loadRegDataSet(config.getString("input.trainSet"),dataSetType,true);
        RegDataSet testSet = TRECFormat.loadRegDataSet(config.getString("input.testSet"),dataSetType,true);

        LinearRegression linearRegression = new LinearRegression(trainSet.getNumFeatures());
        ElasticNetLinearRegOptimizer optimizer = new ElasticNetLinearRegOptimizer(linearRegression,trainSet);
        optimizer.setRegularization(config.getDouble("regularization"));
        optimizer.setL1Ratio(config.getDouble("l1Ratio"));
        System.out.println("before training");
        System.out.println("training set RMSE = "+ RMSE.rmse(linearRegression,trainSet));
        System.out.println("test set RMSE = "+ RMSE.rmse(linearRegression,testSet));
        System.out.println("start training");
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        optimizer.optimize();
        System.out.println("training done");
        System.out.println("time spent on training = "+stopWatch);
        System.out.println("after training");
        System.out.println("training set RMSE = "+ RMSE.rmse(linearRegression,trainSet));
        System.out.println("test set RMSE = "+ RMSE.rmse(linearRegression,testSet));

        System.out.println("number of non-zeros weights in linear regression (not including bias) = "+linearRegression.getWeights().getWeightsWithoutBias().getNumNonZeroElements());


        List<Pair<Integer,Double>> sorted = new ArrayList<>();
        for (Vector.Element element: linearRegression.getWeights().getWeightsWithoutBias().nonZeroes()){
            sorted.add(new Pair<>(element.index(),element.get()));
        }


        Comparator<Pair<Integer, Double>> comparatorByIndex = Comparator.comparing(pair -> pair.getFirst());
        sorted = sorted.stream().sorted(comparatorByIndex).collect(Collectors.toList());

        StringBuilder sb1 = new StringBuilder();
        for (Pair<Integer, Double> pair: sorted){
            int index = pair.getFirst();
            sb1.append(index).append("(").append(trainSet.getFeatureList().get(index).getName()).append(")").append(":").append(pair.getSecond()).append("\n");
        }
        FileUtils.writeStringToFile(new File(output, "features_sorted_by_indices.txt"), sb1.toString());

        System.out.println("all selected features (sorted by indices) are saved to "+new File(output, "features_sorted_by_indices.txt").getAbsolutePath());


        Comparator<Pair<Integer, Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        sorted = sorted.stream().sorted(comparator.reversed()).collect(Collectors.toList());
        StringBuilder sb = new StringBuilder();
        for (Pair<Integer, Double> pair: sorted){
            int index = pair.getFirst();
            sb.append(index).append("(").append(trainSet.getFeatureList().get(index).getName()).append(")").append(":").append(pair.getSecond()).append("\n");
        }
        FileUtils.writeStringToFile(new File(output, "features_sorted_by_weights.txt"), sb.toString());
        System.out.println("all selected features (sorted by absolute weights) are saved to "+new File(output, "features_sorted_by_weights.txt").getAbsolutePath());

        File reportFile = new File(output, "test_predictions.txt");
        report(linearRegression, testSet, reportFile);
        System.out.println("predictions on the test set are written to "+reportFile.getAbsolutePath());

    }

    private static void report(LinearRegression regression, RegDataSet dataSet, File reportFile) throws IOException {
        double[] prediction = regression.predict(dataSet);
        String str = PrintUtil.toMutipleLines(prediction);
        FileUtils.writeStringToFile(reportFile, str);
    }
}
