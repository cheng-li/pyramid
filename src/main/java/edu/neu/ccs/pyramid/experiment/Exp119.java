package edu.neu.ccs.pyramid.experiment;


import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBConfig;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostTrainer;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.ProbRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.List;

/**
 * LS BOOST, soft tree vs hard tree vs hybrid tree, fiji
 * Created by chengli on 6/3/15.
 */
public class Exp119 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        File outputFolder = new File(config.getString("output.folder"));
        outputFolder.mkdirs();
        FileUtils.cleanDirectory(outputFolder);
        train_hard(config);
        train_expectation(config);
        train_hybrid(config);
    }

    static void train_hard(Config config) throws Exception{
        File outputFolder = new File(config.getString("output.folder"),"hard_tree");
        File inputFolder = new File(config.getString("input.folder"));
        RegDataSet dataSet = TRECFormat.loadRegDataSet(new File(inputFolder, "train.trec"),
                DataSetType.REG_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        RegDataSet testSet = TRECFormat.loadRegDataSet(new File(inputFolder, "test.trec"),
                DataSetType.REG_SPARSE, true);

        LSBoost boost = new LSBoost();

        LSBConfig trainConfig = new LSBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .softTreeEarlyStop(config.getBoolean("softTreeEarlyStop"))
                .considerHardTree(true)
                .considerExpectationTree(false)
                .considerProbabilisticTree(false)
                .build();

        LSBoostTrainer trainer = new LSBoostTrainer(boost,trainConfig);

        File trainFile = new File(outputFolder,"train_acc");
        File testFile = new File(outputFolder,"test_acc");
        File typeFile = new File(outputFolder,"type");


        for (int i=0;i<config.getInt("iterations");i++){
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            System.out.println("iteration "+i);
            System.out.println("mse = "+ MSE.mse(boost, dataSet));

            System.out.println("time spent on mse = "+stopWatch);

            trainer.iterate();
            System.out.println("time spent on one iteration = "+stopWatch);

            FileUtils.writeStringToFile(trainFile, "" + MSE.mse(boost, dataSet) + "\n", true);
            FileUtils.writeStringToFile(testFile,""+MSE.mse(boost, testSet)+"\n",true);
            List<Regressor> regressors = boost.getRegressors();
            Regressor regressor = regressors.get(i);
            if (regressor instanceof RegressionTree){
                FileUtils.writeStringToFile(typeFile,"hard tree"+"\n",true);

            }
            if (regressor instanceof ProbRegStump){
                ProbRegStump probRegStump = (ProbRegStump)regressor;
                if (probRegStump.getLossType()== ProbRegStumpTrainer.LossType.SquaredLossOfExpectation){
                    FileUtils.writeStringToFile(typeFile,"expectation tree"+"\n",true);
                }
                if (probRegStump.getLossType()== ProbRegStumpTrainer.LossType.ExpectationOfSquaredLoss){
                    FileUtils.writeStringToFile(typeFile,"probabilistic tree"+"\n",true);
                }
            }
        }

    }


    static void train_hybrid(Config config) throws Exception{
        File outputFolder = new File(config.getString("output.folder"),"hybrid_tree");
        File inputFolder = new File(config.getString("input.folder"));
        RegDataSet dataSet = TRECFormat.loadRegDataSet(new File(inputFolder, "train.trec"),
                DataSetType.REG_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        RegDataSet testSet = TRECFormat.loadRegDataSet(new File(inputFolder, "test.trec"),
                DataSetType.REG_SPARSE, true);

        LSBoost boost = new LSBoost();

        LSBConfig trainConfig = new LSBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .softTreeEarlyStop(config.getBoolean("softTreeEarlyStop"))
                .considerHardTree(true)
                .considerExpectationTree(true)
                .considerProbabilisticTree(false)
                .build();

        LSBoostTrainer trainer = new LSBoostTrainer(boost,trainConfig);

        File trainFile = new File(outputFolder,"train_acc");
        File testFile = new File(outputFolder,"test_acc");
        File typeFile = new File(outputFolder,"type");

        for (int i=0;i<config.getInt("iterations");i++){
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            System.out.println("iteration "+i);
            System.out.println("mse = " + MSE.mse(boost, dataSet));

            System.out.println("time spent on mse = "+stopWatch);

            trainer.iterate();
            System.out.println("time spent on one iteration = "+stopWatch);

            FileUtils.writeStringToFile(trainFile,""+MSE.mse(boost, dataSet)+"\n",true);
            FileUtils.writeStringToFile(testFile,""+MSE.mse(boost, testSet)+"\n",true);
            List<Regressor> regressors = boost.getRegressors();
            Regressor regressor = regressors.get(i);
            if (regressor instanceof RegressionTree){
                FileUtils.writeStringToFile(typeFile,"hard tree"+", ",true);

            }
            if (regressor instanceof ProbRegStump){
                ProbRegStump probRegStump = (ProbRegStump)regressor;
                if (probRegStump.getLossType()== ProbRegStumpTrainer.LossType.SquaredLossOfExpectation){
                    FileUtils.writeStringToFile(typeFile,"expectation tree"+", ",true);
                }
                if (probRegStump.getLossType()== ProbRegStumpTrainer.LossType.ExpectationOfSquaredLoss){
                    FileUtils.writeStringToFile(typeFile,"probabilistic tree"+", ",true);
                }
            }
        }

    }

    static void train_expectation(Config config) throws Exception{
        File outputFolder = new File(config.getString("output.folder"),"expectation_tree");
        File inputFolder = new File(config.getString("input.folder"));
        RegDataSet dataSet = TRECFormat.loadRegDataSet(new File(inputFolder, "train.trec"),
                DataSetType.REG_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        RegDataSet testSet = TRECFormat.loadRegDataSet(new File(inputFolder, "test.trec"),
                DataSetType.REG_SPARSE, true);

        LSBoost boost = new LSBoost();

        LSBConfig trainConfig = new LSBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(0.1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .softTreeEarlyStop(config.getBoolean("softTreeEarlyStop"))
                .considerHardTree(false)
                .considerExpectationTree(true)
                .considerProbabilisticTree(false)
                .build();

        LSBoostTrainer trainer = new LSBoostTrainer(boost,trainConfig);

        File trainFile = new File(outputFolder,"train_acc");
        File testFile = new File(outputFolder,"test_acc");
        File typeFile = new File(outputFolder,"type");

        for (int i=0;i<config.getInt("iterations");i++){
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            System.out.println("iteration "+i);
            System.out.println("mse = "+ MSE.mse(boost, dataSet));

            System.out.println("time spent on mse = "+stopWatch);

            trainer.iterate();
            System.out.println("time spent on one iteration = " + stopWatch);


            FileUtils.writeStringToFile(trainFile,""+MSE.mse(boost, dataSet)+"\n",true);
            FileUtils.writeStringToFile(testFile,""+MSE.mse(boost, testSet)+"\n",true);
            List<Regressor> regressors = boost.getRegressors();
            Regressor regressor = regressors.get(i);
            if (regressor instanceof RegressionTree){
                FileUtils.writeStringToFile(typeFile,"hard tree"+"\n",true);

            }
            if (regressor instanceof ProbRegStump){
                ProbRegStump probRegStump = (ProbRegStump)regressor;
                if (probRegStump.getLossType()== ProbRegStumpTrainer.LossType.SquaredLossOfExpectation){
                    FileUtils.writeStringToFile(typeFile,"expectation tree"+"\n",true);
                }
                if (probRegStump.getLossType()== ProbRegStumpTrainer.LossType.ExpectationOfSquaredLoss){
                    FileUtils.writeStringToFile(typeFile,"probabilistic tree"+"\n",true);
                }
            }
        }

    }
}
