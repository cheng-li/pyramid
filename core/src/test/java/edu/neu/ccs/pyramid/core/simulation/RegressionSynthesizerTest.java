package edu.neu.ccs.pyramid.core.simulation;

import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.RegDataSet;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.Configuration;
import org.apache.logging.log4j.core.config.LoggerConfig;

import java.io.File;

public class RegressionSynthesizerTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();
//        test1();
    }


//    private static void test1() throws Exception{
////        RegDataSet trainSet = RegressionSynthesizer.univarStep();
////        RegDataSet testSet = RegressionSynthesizer.univarStep();
//
////        RegDataSet trainSet = RegressionSynthesizer.univarSine();
////        RegDataSet testSet = RegressionSynthesizer.univarSine();
//
////        RegDataSet trainSet = RegressionSynthesizer.univarLine();
////        RegDataSet testSet = RegressionSynthesizer.univarLine();
//
////        RegDataSet trainSet = RegressionSynthesizer.univarQuadratic();
////        RegDataSet testSet = RegressionSynthesizer.univarQuadratic();
//
//        RegressionSynthesizer regressionSynthesizer = RegressionSynthesizer.getBuilder().build();
//
//        RegDataSet trainSet = regressionSynthesizer.univarExp();
//        RegDataSet testSet = regressionSynthesizer.univarExp();
//
//        TRECFormat.save(trainSet,new File(TMP,"train.trec"));
//        TRECFormat.save(testSet,new File(TMP,"test.trec"));
//
//        int[] activeFeatures = IntStream.range(0, trainSet.getNumFeatures()).toArray();
//        int[] activeDataPoints = IntStream.range(0, trainSet.getNumDataPoints()).toArray();
//        RegTreeConfig regTreeConfig = new RegTreeConfig();
//
//
//        regTreeConfig.setMaxNumLeaves(2);
//        regTreeConfig.setMinDataPerLeaf(5);
//
//
//        regTreeConfig.setNumSplitIntervals(1000);
//        RegressionTree tree = RegTreeTrainer.fit(regTreeConfig,trainSet);
//        System.out.println(tree.toString());
//
//
//        System.out.println("hard rt");
//        System.out.println("training mse = "+ MSE.mse(tree,trainSet));
//        System.out.println("test mse = "+ MSE.mse(tree,testSet));
//
//        String hardTrainPrediction = Arrays.toString(tree.predict(trainSet)).replace("[","").replace("]","");
//        FileUtils.writeStringToFile(new File(TMP,"hardTrainPrediction"),hardTrainPrediction);
//        FileUtils.writeStringToFile(new File(TMP,"hardTrainMSE"),""+MSE.mse(tree,trainSet));
//
//
//        String hardTestPrediction = Arrays.toString(tree.predict(testSet)).replace("[","").replace("]","");
//        FileUtils.writeStringToFile(new File(TMP,"hardTestPrediction"),hardTestPrediction);
//        FileUtils.writeStringToFile(new File(TMP,"hardTestMSE"),""+MSE.mse(tree,testSet));
//
//        SoftRegStumpTrainer trainer = SoftRegStumpTrainer.getBuilder()
//                .setDataSet(trainSet)
//                .setLabels(trainSet.getLabels())
//                .setFeatureType(SoftRegStumpTrainer.FeatureType.FOLLOW_HARD_TREE_FEATURE)
//                .setLossType(SoftRegStumpTrainer.LossType.SquaredLossOfExpectation)
//                .build();
//
//        Optimizer optimizer = trainer.getOptimizer();
//        optimizer.getTerminator().setMode(Terminator.Mode.STANDARD);
//        optimizer.getTerminator().setMaxIteration(10000);
//
//        SoftRegStump softRegStump = trainer.train();
//        System.out.println("prob rt");
//        System.out.println("training mse = "+ MSE.mse(softRegStump,trainSet));
//        System.out.println("test mse = "+ MSE.mse(softRegStump,testSet));
//        System.out.println(softRegStump.toString());
//
//
//        String softTrainPrediction = Arrays.toString(softRegStump.predict(trainSet)).replace("[","").replace("]","");
//        FileUtils.writeStringToFile(new File(TMP,"softTrainPrediction"),softTrainPrediction);
//        FileUtils.writeStringToFile(new File(TMP,"softTrainMSE"),""+MSE.mse(softRegStump,trainSet));
//
//
//        String softTestPrediction = Arrays.toString(softRegStump.predict(testSet)).replace("[","").replace("]","");
//        FileUtils.writeStringToFile(new File(TMP,"softTestPrediction"),softTestPrediction);
//        FileUtils.writeStringToFile(new File(TMP,"softTestMSE"),""+MSE.mse(softRegStump,testSet));
//
//        StringBuilder sb = new StringBuilder();
//        sb.append(((Sigmoid) softRegStump.getGatingFunction()).getWeights().get(0));
//        sb.append(",");
//        sb.append(((Sigmoid) softRegStump.getGatingFunction()).getBias());
//        sb.append(",");
//        sb.append(softRegStump.getLeftOutput());
//        sb.append(",");
//        sb.append(softRegStump.getRightOutput());
//
//        FileUtils.writeStringToFile(new File(TMP,"curve"),sb.toString());
//
//    }

    private static void test2(){
        RegressionSynthesizer regressionSynthesizer = RegressionSynthesizer.getBuilder()
                .setNumDataPoints(100)
                .build();

        //        RegDataSet trainSet = RegressionSynthesizer.univarStep();
//        RegDataSet testSet = RegressionSynthesizer.univarStep();

//        RegDataSet trainSet = RegressionSynthesizer.univarSine();
//        RegDataSet testSet = RegressionSynthesizer.univarSine();

//        RegDataSet trainSet = RegressionSynthesizer.univarLine();
//        RegDataSet testSet = RegressionSynthesizer.univarLine();

//        RegDataSet trainSet = RegressionSynthesizer.univarQuadratic();
//        RegDataSet testSet = RegressionSynthesizer.univarQuadratic();

        RegDataSet trainSet = regressionSynthesizer.univarExp();
        RegDataSet testSet = regressionSynthesizer.univarExp();

        TRECFormat.save(trainSet,new File(TMP,"train.trec"));
        TRECFormat.save(testSet,new File(TMP,"test.trec"));
    }

}