package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBConfig;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostTrainer;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.List;
import java.util.stream.IntStream;

/**
 * follow exp119, check why soft tree performs worse on certain iterations
 * Created by chengli on 6/9/15.
 */
public class Exp122 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        File outputFolder = new File(config.getString("output.folder"));
        outputFolder.mkdirs();
        FileUtils.cleanDirectory(outputFolder);
        train_hybrid(config);
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

        LSBConfig trainConfig = new LSBConfig.Builder()
                .learningRate(config.getDouble("learningRate"))
                .build();

        LSBoostTrainer trainer = new LSBoostTrainer(boost,trainConfig,dataSet);

        File trainFile = new File(outputFolder,"train_per");
        File testFile = new File(outputFolder,"test_per");
        File typeFile = new File(outputFolder,"type");

        List<Integer> iterationsToCheck = config.getIntegers("iterationsToCheck");

        for (int i=0;i<config.getInt("iterations");i++){
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            System.out.println("iteration "+i);


            if (i==0){
                trainer.addPriorRegressor();
            } else {
                if (iterationsToCheck.contains(i)){
                    double[] gradients = trainer.getGradients();
                    RegDataSet subTask = RegDataSetBuilder.getBuilder()
                            .numDataPoints(dataSet.getNumDataPoints())
                            .numFeatures(1)
                            .dense(false)
                            .build();
                    for (int a=0;a<subTask.getNumDataPoints();a++){
                        subTask.setLabel(a,gradients[a]);

                    }
                    int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
                    int[] activeDataPoints = IntStream.range(0, dataSet.getNumDataPoints()).toArray();
                    RegTreeConfig regTreeConfig = new RegTreeConfig();


                    regTreeConfig.setMaxNumLeaves(2);
                    regTreeConfig.setMinDataPerLeaf(1);


                    regTreeConfig.setNumSplitIntervals(1000);
                    RegressionTree tree = RegTreeTrainer.fit(regTreeConfig, dataSet,gradients);
                    int pickedFeature = tree.getRoot().getFeatureIndex();

                    Vector column = dataSet.getColumn(pickedFeature);
                    for (Vector.Element element: column.nonZeroes()){
                        int b = element.index();
                        double v = element.get();
                        subTask.setFeatureValue(b,0,v);
                    }

                    TRECFormat.save(subTask,new File(outputFolder,""+i+".trec"));

                }


                trainer.iterate();

            }
            System.out.println("time spent on one iteration = "+stopWatch);

            FileUtils.writeStringToFile(trainFile,""+ RMSE.rmse(boost, dataSet)+"\n",true);
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


}
