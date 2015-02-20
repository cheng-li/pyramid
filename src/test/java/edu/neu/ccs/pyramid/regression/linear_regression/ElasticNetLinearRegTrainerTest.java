package edu.neu.ccs.pyramid.regression.linear_regression;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.StandardFormat;
import edu.neu.ccs.pyramid.eval.MSE;

import java.io.File;

import static org.junit.Assert.*;

public class ElasticNetLinearRegTrainerTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        RegDataSet dataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/train_data.txt"),
                new File(DATASETS, "spam/train_label.txt"), ",", DataSetType.REG_DENSE, false);
        double[] labels = dataSet.getLabels();
        RegDataSet testDataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/test_data.txt"),
                new File(DATASETS, "spam/test_label.txt"), ",", DataSetType.REG_DENSE,false);
        LinearRegression linearRegression = new LinearRegression(dataSet.getNumFeatures());
        ElasticNetLinearRegTrainer trainer = ElasticNetLinearRegTrainer.getBuilder()
                .setRegularization(0.1).setL1Ratio(0.1).setEpsilon(0.001)
                .build();
        double[] instanceWeights = new double[dataSet.getNumDataPoints()];
        double weight = 1.0/dataSet.getNumDataPoints();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            instanceWeights[i] = weight;
        }
        System.out.println("mse before training = "+ MSE.mse(linearRegression,dataSet));
        trainer.train(linearRegression,dataSet,labels,instanceWeights);
        System.out.println("mse after training = "+ MSE.mse(linearRegression,dataSet));
        System.out.println("test mse after training = "+ MSE.mse(linearRegression,testDataSet));
        System.out.println("non-zeros = "+linearRegression.getWeights().getWeightsWithoutBias().getNumNonZeroElements());
    }

}