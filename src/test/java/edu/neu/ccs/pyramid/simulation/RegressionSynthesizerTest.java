package edu.neu.ccs.pyramid.simulation;

import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;

import java.util.stream.IntStream;

import static org.junit.Assert.*;

public class RegressionSynthesizerTest {
    public static void main(String[] args) {
        test1();
    }


    private static void test1(){
        RegDataSet dataSet = RegressionSynthesizer.univarStepFunction();
        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0, dataSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setActiveFeatures(activeFeatures);

        regTreeConfig.setMaxNumLeaves(2);
        regTreeConfig.setMinDataPerLeaf(5);
        regTreeConfig.setActiveDataPoints(activeDataPoints);

        regTreeConfig.setNumSplitIntervals(1000);
        RegressionTree tree = RegTreeTrainer.fit(regTreeConfig,dataSet);
        System.out.println(tree.toString());

        RegDataSet testSet = RegressionSynthesizer.univarStepFunction();
        System.out.println("training mse = "+ MSE.mse(tree,dataSet));
        System.out.println("test mse = "+ MSE.mse(tree,testSet));
    }

}