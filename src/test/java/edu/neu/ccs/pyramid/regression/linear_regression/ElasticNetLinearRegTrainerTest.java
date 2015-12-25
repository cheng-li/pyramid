package edu.neu.ccs.pyramid.regression.linear_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.StandardFormat;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.util.Grid;

import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;


public class ElasticNetLinearRegTrainerTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
        test2();
    }

    private static void test1() throws Exception{
        RegDataSet dataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/train_data.txt"),
                new File(DATASETS, "spam/train_label.txt"), ",", DataSetType.REG_DENSE, false);
        double[] labels = dataSet.getLabels();
        RegDataSet testDataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/test_data.txt"),
                new File(DATASETS, "spam/test_label.txt"), ",", DataSetType.REG_DENSE,false);
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures());
        ElasticNetLinearRegTrainer trainer = ElasticNetLinearRegTrainer.getBuilder()
                .setRegularization(0.1).setL1Ratio(0.5).setEpsilon(0.001)
                .build();
        double[] instanceWeights = new double[dataSet.getNumDataPoints()];
        double weight = 1.0/dataSet.getNumDataPoints();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            instanceWeights[i] = weight;
        }
        System.out.println("rmse before training = "+ RMSE.rmse(linearRegression, dataSet));
        trainer.train(linearRegression,dataSet,labels,instanceWeights);
        System.out.println("rmse after training = "+ RMSE.rmse(linearRegression, dataSet));
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
        double[] instanceWeights = new double[dataSet.getNumDataPoints()];
        double weight = 1.0/dataSet.getNumDataPoints();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            instanceWeights[i] = weight;
        }
        Comparator<Double> comparator = Comparator.comparing(Double::doubleValue);
        List<Double> grid = Grid.logUniform(0.01, 100, 5).stream().sorted(comparator.reversed()).collect(Collectors.toList());
        List<LinearRegression> regressions = new ArrayList<>();

        for (double regularization: grid){
            ElasticNetLinearRegTrainer trainer = ElasticNetLinearRegTrainer.getBuilder()
                    .setRegularization(regularization).setL1Ratio(0.5).setEpsilon(0.001)
                    .build();
            trainer.train(linearRegression,dataSet,labels,instanceWeights);
            regressions.add(linearRegression.deepCopy());
        }


        for (int i=0;i<grid.size();i++){
            System.out.println("regularization = "+grid.get(i));
            System.out.println("non-zeros = " + regressions.get(i).getWeights().getWeightsWithoutBias().getNumNonZeroElements());
            System.out.println("test rmse  = "+ RMSE.rmse(regressions.get(i), testDataSet));
        }

    }

}