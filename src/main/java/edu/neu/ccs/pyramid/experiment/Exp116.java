package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.lkboost.LKTBConfig;
import edu.neu.ccs.pyramid.classification.lkboost.LKTBTrainer;
import edu.neu.ccs.pyramid.classification.lkboost.LKTreeBoost;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.List;

/**
 * hard tree vs expectation tree vs hybrid tree, accuracy test, on fiji, no newton, no shrinkage
 * Created by chengli on 5/30/15.
 */
public class Exp116 {

    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
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
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(inputFolder, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(inputFolder, "test.trec"),
                DataSetType.CLF_SPARSE, true);

        LKTreeBoost lkTreeBoost = new LKTreeBoost(dataSet.getNumClasses());

        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .setLeafOutputType(LeafOutputType.AVERAGE)
                .considerHardTree(true)
                .considerExpectationTree(false)
                .considerProbabilisticTree(false)
                .build();

        LKTBTrainer lktbTrainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        File trainFile = new File(outputFolder,"train_acc");
        File testFile = new File(outputFolder,"test_acc");
        File typeFile = new File(outputFolder,"type");

        for (int i=0;i<100;i++){
            System.out.println("iteration "+i);
            System.out.println("boosting accuracy = "+ Accuracy.accuracy(lkTreeBoost, dataSet));

            lktbTrainer.iterate();


            FileUtils.writeStringToFile(trainFile,""+Accuracy.accuracy(lkTreeBoost, dataSet)+"\n",true);
            FileUtils.writeStringToFile(testFile,""+Accuracy.accuracy(lkTreeBoost, testSet)+"\n",true);
            List<Regressor> regressors = lkTreeBoost.getRegressors(0);
            Regressor regressor = regressors.get(i);
            if (regressor instanceof RegressionTree){
                FileUtils.writeStringToFile(typeFile,"hard tree"+"\n",true);

            }
            if (regressor instanceof SoftRegStump){
                SoftRegStump softRegStump = (SoftRegStump)regressor;
                if (softRegStump.getLossType()== SoftRegStumpTrainer.LossType.SquaredLossOfExpectation){
                    FileUtils.writeStringToFile(typeFile,"expectation tree"+"\n",true);
                }
                if (softRegStump.getLossType()== SoftRegStumpTrainer.LossType.ExpectationOfSquaredLoss){
                    FileUtils.writeStringToFile(typeFile,"probabilistic tree"+"\n",true);
                }
            }
        }

    }


    static void train_hybrid(Config config) throws Exception{
        File outputFolder = new File(config.getString("output.folder"),"hybrid_tree");
        File inputFolder = new File(config.getString("input.folder"));
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(inputFolder, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(inputFolder, "test.trec"),
                DataSetType.CLF_SPARSE, true);

        LKTreeBoost lkTreeBoost = new LKTreeBoost(dataSet.getNumClasses());

        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .setLeafOutputType(LeafOutputType.AVERAGE)
                .considerHardTree(true)
                .considerExpectationTree(true)
                .considerProbabilisticTree(false)
                .build();

        LKTBTrainer lktbTrainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        File trainFile = new File(outputFolder,"train_acc");
        File testFile = new File(outputFolder,"test_acc");
        File typeFile = new File(outputFolder,"type");

        for (int i=0;i<100;i++){
            System.out.println("iteration "+i);
            System.out.println("boosting accuracy = "+ Accuracy.accuracy(lkTreeBoost, dataSet));

            lktbTrainer.iterate();


            FileUtils.writeStringToFile(trainFile,""+Accuracy.accuracy(lkTreeBoost, dataSet)+"\n",true);
            FileUtils.writeStringToFile(testFile,""+Accuracy.accuracy(lkTreeBoost, testSet)+"\n",true);
            List<Regressor> regressors = lkTreeBoost.getRegressors(0);
            Regressor regressor = regressors.get(i);
            if (regressor instanceof RegressionTree){
                FileUtils.writeStringToFile(typeFile,"hard tree"+", ",true);

            }
            if (regressor instanceof SoftRegStump){
                SoftRegStump softRegStump = (SoftRegStump)regressor;
                if (softRegStump.getLossType()== SoftRegStumpTrainer.LossType.SquaredLossOfExpectation){
                    FileUtils.writeStringToFile(typeFile,"expectation tree"+", ",true);
                }
                if (softRegStump.getLossType()== SoftRegStumpTrainer.LossType.ExpectationOfSquaredLoss){
                    FileUtils.writeStringToFile(typeFile,"probabilistic tree"+", ",true);
                }
            }
        }

    }

    static void train_expectation(Config config) throws Exception{
        File outputFolder = new File(config.getString("output.folder"),"expectation_tree");
        File inputFolder = new File(config.getString("input.folder"));
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(inputFolder, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(inputFolder, "test.trec"),
                DataSetType.CLF_SPARSE, true);

        LKTreeBoost lkTreeBoost = new LKTreeBoost(dataSet.getNumClasses());

        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .setLeafOutputType(LeafOutputType.AVERAGE)
                .considerHardTree(false)
                .considerExpectationTree(true)
                .considerProbabilisticTree(false)
                .build();

        LKTBTrainer lktbTrainer = new LKTBTrainer(trainConfig,lkTreeBoost);

        File trainFile = new File(outputFolder,"train_acc");
        File testFile = new File(outputFolder,"test_acc");
        File typeFile = new File(outputFolder,"type");

        for (int i=0;i<100;i++){
            System.out.println("iteration "+i);
            System.out.println("boosting accuracy = "+ Accuracy.accuracy(lkTreeBoost, dataSet));

            lktbTrainer.iterate();


            FileUtils.writeStringToFile(trainFile,""+Accuracy.accuracy(lkTreeBoost, dataSet)+"\n",true);
            FileUtils.writeStringToFile(testFile,""+Accuracy.accuracy(lkTreeBoost, testSet)+"\n",true);
            List<Regressor> regressors = lkTreeBoost.getRegressors(0);
            Regressor regressor = regressors.get(i);
            if (regressor instanceof RegressionTree){
                FileUtils.writeStringToFile(typeFile,"hard tree"+"\n",true);

            }
            if (regressor instanceof SoftRegStump){
                SoftRegStump softRegStump = (SoftRegStump)regressor;
                if (softRegStump.getLossType()== SoftRegStumpTrainer.LossType.SquaredLossOfExpectation){
                    FileUtils.writeStringToFile(typeFile,"expectation tree"+"\n",true);
                }
                if (softRegStump.getLossType()== SoftRegStumpTrainer.LossType.ExpectationOfSquaredLoss){
                    FileUtils.writeStringToFile(typeFile,"probabilistic tree"+"\n",true);
                }
            }
        }

    }
}
