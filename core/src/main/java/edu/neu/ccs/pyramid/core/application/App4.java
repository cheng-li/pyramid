package edu.neu.ccs.pyramid.core.application;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import edu.neu.ccs.pyramid.core.eval.RMSE;
import edu.neu.ccs.pyramid.core.regression.linear_regression.ElasticNetLinearRegOptimizer;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.dataset.RegDataSet;
import edu.neu.ccs.pyramid.core.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.core.util.Pair;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Linear Regression with Elasticnet (L1+L2) regularization
 * Created by chengli on 2/25/16.
 */
public class App4 {

    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

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
        System.out.println("all non-zero weights in the format of (feature index:feature weight) pairs:");
        System.out.println("(Note that feature indices start from 0)");
        System.out.println(linearRegression.getWeights().getWeightsWithoutBias());

        List<Pair<Integer,Double>> sorted = new ArrayList<>();
        for (Vector.Element element: linearRegression.getWeights().getWeightsWithoutBias().nonZeroes()){
            sorted.add(new Pair<>(element.index(),element.get()));
        }

        Comparator<Pair<Integer, Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        sorted = sorted.stream().sorted(comparator.reversed()).collect(Collectors.toList());
        System.out.println("all non-zero weights in sorted order:");
        StringBuilder sb = new StringBuilder();
        for (Pair<Integer, Double> pair: sorted){
            sb.append(pair.getFirst()).append(":").append(pair.getSecond()).append(", ");
        }
        System.out.println(sb.toString());

    }
}
