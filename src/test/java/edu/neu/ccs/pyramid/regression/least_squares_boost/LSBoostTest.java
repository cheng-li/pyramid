package edu.neu.ccs.pyramid.regression.least_squares_boost;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.StandardFormat;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.regression.linear_regression.ElasticNetLinearRegTrainer;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.simulation.RegressionSynthesizer;

import java.io.File;
import java.util.stream.IntStream;

public class LSBoostTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test2();
    }



    private static void test1() throws Exception{

        RegressionSynthesizer regressionSynthesizer = RegressionSynthesizer.getBuilder().build();

        RegDataSet trainSet = regressionSynthesizer.univarStep();
        RegDataSet testSet = regressionSynthesizer.univarStep();

//        RegDataSet trainSet = regressionSynthesizer.univarSine();
//        RegDataSet testSet = regressionSynthesizer.univarSine();

//        RegDataSet trainSet = regressionSynthesizer.univarLine();
//        RegDataSet testSet = regressionSynthesizer.univarLine();

//        RegDataSet trainSet = regressionSynthesizer.univarQuadratic();
//        RegDataSet testSet = regressionSynthesizer.univarQuadratic();


//        RegDataSet trainSet = regressionSynthesizer.univarExp();
//        RegDataSet testSet = regressionSynthesizer.univarExp();
        int[] activeFeatures = IntStream.range(0, trainSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0, trainSet.getNumDataPoints()).toArray();

        LSBConfig lsbConfig = LSBConfig.getBuilder()

                .learningRate(1)

                .build();

        LSBoost lsBoost = new LSBoost();
        LSBoostTrainer trainer = new LSBoostTrainer(lsBoost,lsbConfig,trainSet);
        trainer.addPriorRegressor();
        for (int i=0;i<100;i++){
            System.out.println("iteration "+i);
            System.out.println("train MSE = "+ MSE.mse(lsBoost,trainSet));
            System.out.println("test MSE = "+ MSE.mse(lsBoost,testSet));
            trainer.iterate();
        }

    }

    private static void test2() throws Exception{
        RegDataSet trainSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/train_data.txt"),
                new File(DATASETS, "spam/train_label.txt"), ",", DataSetType.REG_DENSE, false);

        RegDataSet testSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/test_data.txt"),
                new File(DATASETS, "spam/test_label.txt"), ",", DataSetType.REG_DENSE,false);
        LSBConfig lsbConfig = LSBConfig.getBuilder()

                .learningRate(1)

                .build();

        LSBoost lsBoost = new LSBoost();
        LSBoostTrainer trainer = new LSBoostTrainer(lsBoost,lsbConfig,trainSet);
        trainer.addPriorRegressor();
        for (int i=0;i<100;i++){
            System.out.println("iteration "+i);
            System.out.println("train RMSE = "+ RMSE.rmse(lsBoost, trainSet));
            System.out.println("test RMSE = "+ RMSE.rmse(lsBoost, testSet));
            trainer.iterate();
        }
    }

}