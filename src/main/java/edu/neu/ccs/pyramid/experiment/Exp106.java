package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.optimization.Optimizer;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStump;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStumpTrainer;
import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.Sigmoid;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.simulation.RegressionSynthesizer;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * soft tree, convergence test
 * Created by chengli on 5/24/15.
 */
public class Exp106 {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");


    public static void main(String[] args) throws Exception{
        String[] functionNames = {"1","2","3","4","5","6","7","8"};
        int[] iterations = {10,20,50,100};
        for (String name: functionNames){
            //clear old results
            FileUtils.deleteDirectory(new File(TMP,name));
            for (int iteration: iterations){
                testFunction(name,iteration);
            }

        }

    }


    private static RegDataSet sample(String name){
        RegDataSet dataSet = null;
        RegressionSynthesizer regressionSynthesizer = RegressionSynthesizer.getBuilder().build();
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
    private static void testFunction(String name, int iteration) throws Exception{
        File folder = new File(new File(TMP,name),""+iteration);
        folder.mkdirs();

        File csvFile = new File(new File(TMP,name),"parameters.csv");

        RegDataSet trainSet = sample(name);
        RegDataSet testSet = sample(name);

        TRECFormat.save(trainSet, new File(folder, "train.trec"));
        TRECFormat.save(testSet,new File(folder,"test.trec"));

        int[] activeFeatures = IntStream.range(0, trainSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0, trainSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();


        regTreeConfig.setMaxNumLeaves(2);
        regTreeConfig.setMinDataPerLeaf(1);


        regTreeConfig.setNumSplitIntervals(1000);
        RegressionTree tree = RegTreeTrainer.fit(regTreeConfig, trainSet);
        System.out.println(tree.toString());


        System.out.println("hard rt");
        System.out.println("training mse = "+ MSE.mse(tree, trainSet));
        System.out.println("test mse = "+ MSE.mse(tree,testSet));

        String hardTrainPrediction = Arrays.toString(tree.predict(trainSet)).replace("[","").replace("]","");
        FileUtils.writeStringToFile(new File(folder, "hardTrainPrediction"), hardTrainPrediction);
        FileUtils.writeStringToFile(new File(folder,"hardTrainMSE"),""+MSE.mse(tree,trainSet));


        String hardTestPrediction = Arrays.toString(tree.predict(testSet)).replace("[","").replace("]","");
        FileUtils.writeStringToFile(new File(folder,"hardTestPrediction"),hardTestPrediction);
        FileUtils.writeStringToFile(new File(folder,"hardTestMSE"),""+MSE.mse(tree,testSet));

        SoftRegStumpTrainer trainer = SoftRegStumpTrainer.getBuilder()
                .setDataSet(trainSet)
                .setLabels(trainSet.getLabels())
                .setFeatureType(SoftRegStumpTrainer.FeatureType.FOLLOW_HARD_TREE_FEATURE)
                .setLossType(SoftRegStumpTrainer.LossType.SquaredLossOfExpectation)
                .build();

        Optimizer optimizer = trainer.getOptimizer();
        optimizer.setCheckConvergence(false);
        optimizer.setMaxIteration(iteration);
        SoftRegStump softRegStump = trainer.train();
        System.out.println("prob rt");
        System.out.println("training mse = "+ MSE.mse(softRegStump,trainSet));
        System.out.println("test mse = "+ MSE.mse(softRegStump,testSet));
        System.out.println(softRegStump.toString());


        String softTrainPrediction = Arrays.toString(softRegStump.predict(trainSet)).replace("[","").replace("]","");
        FileUtils.writeStringToFile(new File(folder,"softTrainPrediction"),softTrainPrediction);
        FileUtils.writeStringToFile(new File(folder,"softTrainMSE"),""+MSE.mse(softRegStump,trainSet));


        String softTestPrediction = Arrays.toString(softRegStump.predict(testSet)).replace("[","").replace("]","");
        FileUtils.writeStringToFile(new File(folder,"softTestPrediction"),softTestPrediction);
        FileUtils.writeStringToFile(new File(folder,"softTestMSE"),""+MSE.mse(softRegStump,testSet));

        StringBuilder sb = new StringBuilder();
        sb.append(((Sigmoid) softRegStump.getGatingFunction()).getWeights().get(0));
        sb.append(",");
        sb.append(((Sigmoid) softRegStump.getGatingFunction()).getBias());
        sb.append(",");
        sb.append(softRegStump.getLeftOutput());
        sb.append(",");
        sb.append(softRegStump.getRightOutput());
        sb.append("\n");

        FileUtils.writeStringToFile(new File(folder, "curve"), sb.toString());


        if (!csvFile.exists()){
            StringBuilder sb2 = new StringBuilder();
            sb2.append("iteration");
            sb2.append(",");
            sb2.append("a");
            sb2.append(",");
            sb2.append("b");
            sb2.append(",");
            sb2.append("left value");
            sb2.append(",");
            sb2.append("right value");
            sb2.append(",");
            sb2.append("train MSE");
            sb2.append(",");
            sb2.append("test MSE");
            sb2.append("\n");
            FileUtils.writeStringToFile(csvFile, sb2.toString(), true);
        }

        DecimalFormat df = new DecimalFormat("0.00");
        DecimalFormat df2 = new DecimalFormat("0.##E0");
        StringBuilder sb3 = new StringBuilder();
        sb3.append(iteration);
        sb3.append(",");
        sb3.append(df.format(((Sigmoid) softRegStump.getGatingFunction()).getWeights().get(0)));
        sb3.append(",");
        sb3.append(df.format(((Sigmoid) softRegStump.getGatingFunction()).getBias()));
        sb3.append(",");
        sb3.append(df.format(softRegStump.getLeftOutput()));
        sb3.append(",");
        sb3.append(df.format(softRegStump.getRightOutput()));
        sb3.append(",");
        sb3.append(df2.format(MSE.mse(softRegStump, trainSet)));
        sb3.append(",");
        sb3.append(df2.format(MSE.mse(softRegStump, testSet)));
        sb3.append("\n");

        FileUtils.writeStringToFile(csvFile, sb3.toString(), true);




    }
}
