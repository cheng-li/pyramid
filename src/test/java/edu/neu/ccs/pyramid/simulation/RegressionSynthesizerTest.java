package edu.neu.ccs.pyramid.simulation;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.Sigmoid;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.apache.commons.io.FileUtils;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.io.File;
import java.util.Arrays;
import java.util.stream.IntStream;

import static org.junit.Assert.*;

public class RegressionSynthesizerTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();
        test1();
    }


    private static void test1() throws Exception{
//        RegDataSet trainSet = RegressionSynthesizer.univarStep();
//        RegDataSet testSet = RegressionSynthesizer.univarStep();

//        RegDataSet trainSet = RegressionSynthesizer.univarSine();
//        RegDataSet testSet = RegressionSynthesizer.univarSine();

//        RegDataSet trainSet = RegressionSynthesizer.univarLine();
//        RegDataSet testSet = RegressionSynthesizer.univarLine();

//        RegDataSet trainSet = RegressionSynthesizer.univarQuadratic();
//        RegDataSet testSet = RegressionSynthesizer.univarQuadratic();

        RegressionSynthesizer regressionSynthesizer = RegressionSynthesizer.getBuilder().build();

        RegDataSet trainSet = regressionSynthesizer.univarExp();
        RegDataSet testSet = regressionSynthesizer.univarExp();

        TRECFormat.save(trainSet,new File(TMP,"train.trec"));
        TRECFormat.save(testSet,new File(TMP,"test.trec"));

        int[] activeFeatures = IntStream.range(0, trainSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0, trainSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setActiveFeatures(activeFeatures);

        regTreeConfig.setMaxNumLeaves(2);
        regTreeConfig.setMinDataPerLeaf(5);
        regTreeConfig.setActiveDataPoints(activeDataPoints);

        regTreeConfig.setNumSplitIntervals(1000);
        RegressionTree tree = RegTreeTrainer.fit(regTreeConfig,trainSet);
        System.out.println(tree.toString());


        System.out.println("hard rt");
        System.out.println("training mse = "+ MSE.mse(tree,trainSet));
        System.out.println("test mse = "+ MSE.mse(tree,testSet));

        String hardTrainPrediction = Arrays.toString(tree.predict(trainSet)).replace("[","").replace("]","");
        FileUtils.writeStringToFile(new File(TMP,"hardTrainPrediction"),hardTrainPrediction);
        FileUtils.writeStringToFile(new File(TMP,"hardTrainMSE"),""+MSE.mse(tree,trainSet));


        String hardTestPrediction = Arrays.toString(tree.predict(testSet)).replace("[","").replace("]","");
        FileUtils.writeStringToFile(new File(TMP,"hardTestPrediction"),hardTestPrediction);
        FileUtils.writeStringToFile(new File(TMP,"hardTestMSE"),""+MSE.mse(tree,testSet));

        ProbRegStumpTrainer trainer = new ProbRegStumpTrainer(trainSet,trainSet.getLabels(),ProbRegStumpTrainer.FeatureType.ALL_FEATURES);
        LBFGS lbfgs = trainer.getLbfgs();
        lbfgs.setCheckConvergence(true);
        lbfgs.setMaxIteration(10000);
        ProbRegStump probRegStump = trainer.train();
        System.out.println("prob rt");
        System.out.println("training mse = "+ MSE.mse(probRegStump,trainSet));
        System.out.println("test mse = "+ MSE.mse(probRegStump,testSet));
        System.out.println(probRegStump.toString());


        String softTrainPrediction = Arrays.toString(probRegStump.predict(trainSet)).replace("[","").replace("]","");
        FileUtils.writeStringToFile(new File(TMP,"softTrainPrediction"),softTrainPrediction);
        FileUtils.writeStringToFile(new File(TMP,"softTrainMSE"),""+MSE.mse(probRegStump,trainSet));


        String softTestPrediction = Arrays.toString(probRegStump.predict(testSet)).replace("[","").replace("]","");
        FileUtils.writeStringToFile(new File(TMP,"softTestPrediction"),softTestPrediction);
        FileUtils.writeStringToFile(new File(TMP,"softTestMSE"),""+MSE.mse(probRegStump,testSet));

        StringBuilder sb = new StringBuilder();
        sb.append(((Sigmoid)probRegStump.getGatingFunction()).getWeights().get(0));
        sb.append(",");
        sb.append(((Sigmoid)probRegStump.getGatingFunction()).getBias());
        sb.append(",");
        sb.append(probRegStump.getLeftOutput());
        sb.append(",");
        sb.append(probRegStump.getRightOutput());

        FileUtils.writeStringToFile(new File(TMP,"curve"),sb.toString());

    }

}