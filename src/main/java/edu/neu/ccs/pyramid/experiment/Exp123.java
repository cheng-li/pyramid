package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.optimization.*;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStumpTrainer;
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
import java.util.stream.IntStream;

/**
 * follow exp122, soft tree vs hard tree, failure analysis
 * Created by chengli on 6/9/15.
 */
public class Exp123 {
    public static void main(String[] args) throws Exception{
        LoggerContext ctx = (LoggerContext) LogManager.getContext(false);
        Configuration config = ctx.getConfiguration();
        LoggerConfig loggerConfig = config.getLoggerConfig(LogManager.ROOT_LOGGER_NAME);
        loggerConfig.setLevel(Level.DEBUG);
        ctx.updateLoggers();
        String path = "/Users/chengli/tmp/exp122/e2006-tfidf/1/hybrid_tree/1.trec";
        testDataSet(path);
    }

    static void testDataSet(String path) throws Exception{
        RegDataSet trainSet = TRECFormat.loadRegDataSet(path, DataSetType.REG_SPARSE, false);


        int[] activeFeatures = IntStream.range(0, trainSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0, trainSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();


        regTreeConfig.setMaxNumLeaves(2);
        regTreeConfig.setMinDataPerLeaf(1);


        regTreeConfig.setNumSplitIntervals(50);
        RegressionTree hardTree = RegTreeTrainer.fit(regTreeConfig, trainSet);
        System.out.println("hard tree = "+hardTree);
        System.out.println("mse = "+ MSE.mse(hardTree, trainSet));

        double hardMse = MSE.mse(hardTree, trainSet);
        System.out.println(hardMse*0.5*trainSet.getNumDataPoints());




        int[] iterations = {0,10,20,100};
        for (int i: iterations){
            System.out.println("=====================");
            SoftRegStumpTrainer trainer = SoftRegStumpTrainer.getBuilder()
                    .setDataSet(trainSet)
                    .setLabels(trainSet.getLabels())
                    .setFeatureType(SoftRegStumpTrainer.FeatureType.FOLLOW_HARD_TREE_FEATURE)
                    .setLossType(SoftRegStumpTrainer.LossType.SquaredLossOfExpectation)
                    .build();
            Optimizer optimizer = trainer.getOptimizer();
            optimizer.getTerminator().setMode(Terminator.Mode.FINISH_MAX_ITER);
            optimizer.getTerminator().setMaxIteration(i);


            SoftRegStump softTree = trainer.train();
            System.out.println("soft tree = "+softTree);
            System.out.println("mse = "+ MSE.mse(softTree, trainSet));

            StringBuilder sb = new StringBuilder();
            sb.append(((Sigmoid)softTree.getGatingFunction()).getWeights().get(0));
            sb.append(",");
            sb.append(((Sigmoid)softTree.getGatingFunction()).getBias());
            sb.append(",");
            sb.append(softTree.getLeftOutput());
            sb.append(",");
            sb.append(softTree.getRightOutput());

            FileUtils.writeStringToFile(new File(path, "curve"+i), sb.toString());

        }


    }
}
