package edu.neu.ccs.pyramid.regression.least_squares_boost;

import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBTrainer;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.simulation.RegressionSynthesizer;

import java.util.stream.IntStream;

import static org.junit.Assert.*;

public class LSBoostTest {

    public static void main(String[] args) throws Exception{
        test1();
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

        LSBConfig lsbConfig = LSBConfig.getBuilder(trainSet)
                .considerHardTree(true)
                .considerExpectationTree(false)
                .considerProbabilisticTree(false)
                .dataSamplingRate(1)
                .featureSamplingRate(1)
                .learningRate(1)
                .minDataPerLeaf(1)
                .numLeaves(2)
                .numSplitIntervals(100)
                .randomLevel(1)
                .build();

        LSBoost lsBoost = new LSBoost();
        LSBoostTrainer trainer = new LSBoostTrainer(lsBoost,lsbConfig);
        trainer.addPriorRegressor();
        for (int i=0;i<100;i++){
            System.out.println("iteration "+i);
            System.out.println("train MSE = "+ MSE.mse(lsBoost,trainSet));
            System.out.println("test MSE = "+ MSE.mse(lsBoost,testSet));
            trainer.iterate();
        }

    }

}