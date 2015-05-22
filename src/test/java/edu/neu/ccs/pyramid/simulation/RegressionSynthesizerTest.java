package edu.neu.ccs.pyramid.simulation;

import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.util.stream.IntStream;

import static org.junit.Assert.*;

public class RegressionSynthesizerTest {
    public static void main(String[] args) {
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();
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
        System.out.println("hard rt");
        System.out.println("training mse = "+ MSE.mse(tree,dataSet));
        System.out.println("test mse = "+ MSE.mse(tree,testSet));

        ProbRegStump probRegStump = ProbRegStumpTrainer.train(dataSet,dataSet.getLabels());
        System.out.println("prob rt");
        System.out.println("training mse = "+ MSE.mse(probRegStump,dataSet));
        System.out.println("test mse = "+ MSE.mse(probRegStump,testSet));
        System.out.println(probRegStump.toString());

    }

}