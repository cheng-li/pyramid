package edu.neu.ccs.pyramid.core.regression.linear_regression;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.RegDataSet;
import edu.neu.ccs.pyramid.core.eval.RMSE;
import edu.neu.ccs.pyramid.core.simulation.RegressionSynthesizer;
import edu.neu.ccs.pyramid.core.util.Grid;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.dataset.StandardFormat;
import edu.neu.ccs.pyramid.core.util.Pair;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;


public class ElasticNetLinearRegTrainerTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
//        test1();
//        test2();
        test3();
    }

    private static void test1() throws Exception{
        RegDataSet dataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/train_data.txt"),
                new File(DATASETS, "spam/train_label.txt"), ",", DataSetType.REG_DENSE, false);
        double[] labels = dataSet.getLabels();
        RegDataSet testDataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/test_data.txt"),
                new File(DATASETS, "spam/test_label.txt"), ",", DataSetType.REG_DENSE,false);
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures());

        ElasticNetLinearRegOptimizer trainer = new ElasticNetLinearRegOptimizer(linearRegression,dataSet,labels);
        trainer.setRegularization(10);
        trainer.setL1Ratio(0.5);

        System.out.println("train rmse before training = "+ RMSE.rmse(linearRegression, dataSet));
        System.out.println("test rmse before training = "+ RMSE.rmse(linearRegression, testDataSet));
        trainer.optimize();
        System.out.println("train rmse after training = "+ RMSE.rmse(linearRegression, dataSet));
        System.out.println("test rmse after training = "+ RMSE.rmse(linearRegression, testDataSet));
        System.out.println("non-zeros = "+linearRegression.getWeights().getWeightsWithoutBias().getNumNonZeroElements());
    }

    private static void test2() throws Exception{
        RegDataSet dataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/train_data.txt"),
                new File(DATASETS, "spam/train_label.txt"), ",", DataSetType.REG_DENSE, false);
        double[] labels = dataSet.getLabels();
        RegDataSet testDataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/test_data.txt"),
                new File(DATASETS, "spam/test_label.txt"), ",", DataSetType.REG_DENSE, false);
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures());

        Comparator<Double> comparator = Comparator.comparing(Double::doubleValue);
        List<Double> grid = Grid.logUniform(0.01, 100, 5).stream().sorted(comparator.reversed()).collect(Collectors.toList());
        List<LinearRegression> regressions = new ArrayList<>();

        for (double regularization: grid){
            ElasticNetLinearRegOptimizer trainer = new ElasticNetLinearRegOptimizer(linearRegression,dataSet,labels);
            trainer.setRegularization(regularization);
            trainer.setL1Ratio(0.5);
            trainer.optimize();
            regressions.add(linearRegression.deepCopy());
        }


        for (int i=0;i<grid.size();i++){
            System.out.println("regularization = "+grid.get(i));
            System.out.println("non-zeros = " + regressions.get(i).getWeights().getWeightsWithoutBias().getNumNonZeroElements());
            System.out.println("test rmse  = "+ RMSE.rmse(regressions.get(i), testDataSet));
        }

    }

    private static void test3() throws Exception{
        RegDataSet dataSet = RegressionSynthesizer.linear();
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures());

        ElasticNetLinearRegOptimizer trainer = new ElasticNetLinearRegOptimizer(linearRegression,dataSet);
        trainer.setRegularization(0.001);
        trainer.setL1Ratio(0.1);

        System.out.println("train rmse before training = "+ RMSE.rmse(linearRegression, dataSet));

        trainer.optimize();
        System.out.println("train rmse after training = "+ RMSE.rmse(linearRegression, dataSet));
        System.out.println("non-zeros = "+linearRegression.getWeights().getWeightsWithoutBias());
        List<Pair<Integer,Double>> pairs = new ArrayList<>();
        for (Vector.Element element: linearRegression.getWeights().getWeightsWithoutBias().nonZeroes()){
            pairs.add(new Pair<>(element.index(),element.get()));
        }
        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(pair-> pair.getSecond());
        Set<Integer> set = pairs.stream().sorted(comparator.reversed()).limit(4).map(pair-> pair.getFirst()).collect(Collectors.toSet());
        Set<Integer> trueSet = new HashSet<>();
        trueSet.add(0);
        trueSet.add(1);
        trueSet.add(2);
        trueSet.add(3);
        if (set.equals(trueSet)){
            System.out.println("correct");
        } else {
            System.out.println("incorrect");
        }


    }

}