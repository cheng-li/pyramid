package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBConfig;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostTrainer;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.simulation.RegressionSynthesizer;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.List;

/**
 * LS BOOST, soft tree vs hard tree vs hybrid tree, simulation data, local, no shrinkage
 * Created by chengli on 6/3/15.
 */
public class Exp120 {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{


        FileUtils.cleanDirectory(new File(TMP));
        String[] functionNames = {"1","2","3","4","5","6","7","8"};
        for (String name: functionNames){
            RegDataSet dataSet = sample(name);
            RegDataSet testSet = sample(name);
            train_hard(name, dataSet, testSet);
            train_expectation(name, dataSet, testSet);
            train_hybrid(name, dataSet, testSet);
        }


    }

    private static RegDataSet sample(String name){
        RegDataSet dataSet = null;
        RegressionSynthesizer regressionSynthesizer = RegressionSynthesizer.getBuilder()
                .setNumDataPoints(100)
                .setNoiseSD(0.000001).build();
        switch (name) {
            case "1":
                dataSet = regressionSynthesizer.univarStep();
                break;
            case "2":
                dataSet = regressionSynthesizer.univarLine();
                break;
            case "3":
                dataSet = regressionSynthesizer.univarQuadratic();
                break;
            case "4":
                dataSet = regressionSynthesizer.univarExp();
                break;
            case "5":
                dataSet = regressionSynthesizer.univarSine();
                break;
            case "6":
                dataSet = regressionSynthesizer.univarNormal();
                break;
            case "7":
                dataSet = regressionSynthesizer.univarBeta();
                break;
            case "8":
                dataSet = regressionSynthesizer.univarPiecewiseLinear();
                break;
        }

        return dataSet;

    }

    static void train_hard(String name, RegDataSet dataSet, RegDataSet testSet) throws Exception{
        File outputFolder = new File(new File(TMP,name),"hard_tree");
        outputFolder.mkdirs();


        LSBoost boost = new LSBoost();

        LSBConfig trainConfig = new LSBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .softTreeEarlyStop(false)
                .considerHardTree(true)
                .considerExpectationTree(false)
                .considerProbabilisticTree(false)
                .build();

        LSBoostTrainer trainer = new LSBoostTrainer(boost,trainConfig);

        File trainFile = new File(outputFolder,"train_per");
        File testFile = new File(outputFolder,"test_per");
        File typeFile = new File(outputFolder,"type");


        for (int i=0;i<100;i++){
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            System.out.println("iteration "+i);
            
            if (i==0){
                trainer.addPriorRegressor();
            } else {
                trainer.iterate();
            }


            System.out.println("time spent on one iteration = "+stopWatch);

            FileUtils.writeStringToFile(trainFile, "" + RMSE.rmse(boost, dataSet) + "\n", true);
            FileUtils.writeStringToFile(testFile,""+RMSE.rmse(boost, testSet)+"\n",true);
            List<Regressor> regressors = boost.getRegressors();
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


    static void train_hybrid(String name, RegDataSet dataSet, RegDataSet testSet) throws Exception{
        File outputFolder = new File(new File(TMP,name),"hybrid_tree");
        outputFolder.mkdirs();


        LSBoost boost = new LSBoost();

        LSBConfig trainConfig = new LSBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .softTreeEarlyStop(false)
                .considerHardTree(true)
                .considerExpectationTree(true)
                .considerProbabilisticTree(false)
                .build();

        LSBoostTrainer trainer = new LSBoostTrainer(boost,trainConfig);

        File trainFile = new File(outputFolder,"train_per");
        File testFile = new File(outputFolder,"test_per");
        File typeFile = new File(outputFolder,"type");

        for (int i=0;i<100;i++){
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            System.out.println("iteration "+i);


            if (i==0){
                trainer.addPriorRegressor();
            } else {
                trainer.iterate();
            }
            System.out.println("time spent on one iteration = "+stopWatch);

            FileUtils.writeStringToFile(trainFile,""+RMSE.rmse(boost, dataSet)+"\n",true);
            FileUtils.writeStringToFile(testFile,""+RMSE.rmse(boost, testSet)+"\n",true);
            List<Regressor> regressors = boost.getRegressors();
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

    static void train_expectation(String name, RegDataSet dataSet, RegDataSet testSet) throws Exception{
        File outputFolder = new File(new File(TMP,name),"expectation_tree");
        outputFolder.mkdirs();

        LSBoost boost = new LSBoost();

        LSBConfig trainConfig = new LSBConfig.Builder(dataSet)
                .numLeaves(2).learningRate(1).numSplitIntervals(50).minDataPerLeaf(1)
                .dataSamplingRate(1).featureSamplingRate(1)
                .randomLevel(1)
                .softTreeEarlyStop(false)
                .considerHardTree(false)
                .considerExpectationTree(true)
                .considerProbabilisticTree(false)
                .build();

        LSBoostTrainer trainer = new LSBoostTrainer(boost,trainConfig);

        File trainFile = new File(outputFolder,"train_per");
        File testFile = new File(outputFolder,"test_per");
        File typeFile = new File(outputFolder,"type");

        for (int i=0;i<100;i++){
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            System.out.println("iteration "+i);


            if (i==0){
                trainer.addPriorRegressor();
            } else {
                trainer.iterate();
            }
            System.out.println("time spent on one iteration = " + stopWatch);


            FileUtils.writeStringToFile(trainFile,""+RMSE.rmse(boost, dataSet)+"\n",true);
            FileUtils.writeStringToFile(testFile,""+RMSE.rmse(boost, testSet)+"\n",true);
            List<Regressor> regressors = boost.getRegressors();
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
