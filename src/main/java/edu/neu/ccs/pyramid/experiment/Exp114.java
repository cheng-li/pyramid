//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.classification.lkboost.LKTBConfig;
//import edu.neu.ccs.pyramid.classification.lkboost.LKTBTrainer;
//import edu.neu.ccs.pyramid.classification.lkboost.LKTreeBoost;
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.ClfDataSet;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.GradientMatrix;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//import edu.neu.ccs.pyramid.regression.Regressor;
//import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStump;
//import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.SoftRegStumpTrainer;
//import edu.neu.ccs.pyramid.regression.probabilistic_regression_tree.Sigmoid;
//import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
//import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
//import org.apache.commons.io.FileUtils;
//
//import java.io.File;
//import java.util.Arrays;
//import java.util.List;
//
///**
// * visualize residuals for lktb with probabilistic trees
// * Created by chengli on 5/30/15.
// */
//public class Exp114 {
//    private static final Config config = new Config("config/local.config");
//    private static final String DATASETS = config.getString("input.datasets");
//    private static final String TMP = config.getString("output.tmp");
//
//    public static void main(String[] args) throws Exception{
//        File folder = new File(TMP);
//        FileUtils.cleanDirectory(folder);
//        train();
//    }
//
//    static void train() throws Exception{
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
//                DataSetType.CLF_SPARSE, true);
//        System.out.println(dataSet.getMetaInfo());
//
//
//
//        LKTreeBoost lkTreeBoost = new LKTreeBoost(2);
//
//
//        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
//                .numLeaves(2).learningRate(1).numSplitIntervals(50).minDataPerLeaf(1)
//                .dataSamplingRate(1).featureSamplingRate(1)
//                .randomLevel(1)
//                .setLeafOutputType(LeafOutputType.AVERAGE)
//                .considerHardTree(false)
//                .considerExpectationTree(false)
//                .considerProbabilisticTree(true)
//                .build();
//
//        LKTBTrainer lktbTrainer = new LKTBTrainer(trainConfig,lkTreeBoost);
////        lktbTrainer.addLogisticRegression(logisticRegression);
//
//        TRECFormat.save(dataSet, new File(TMP, "train.trec"));
//
//        File gradientsFile = new File(TMP,"gradients");
//
//
//        File featuresFile = new File(TMP,"features");
//
//
//        File thresFile = new File(TMP,"threshold");
//
//
//        File leftFile = new File(TMP,"leftOutput");
//
//        File rightFile = new File(TMP,"rightOutput");
//
//
//        File predFile = new File(TMP,"prediction");
//
//
//        for (int i=0;i<100;i++){
//
//            System.out.println("iteration "+i);
//            System.out.println("boosting accuracy = "+ Accuracy.accuracy(lkTreeBoost, dataSet));
//            GradientMatrix gradientMatrix = lktbTrainer.getGradientMatrix();
//            double[] gradients = gradientMatrix.getGradientsForClass(0);
//            String gradientStr = Arrays.toString(gradients).replace("[","").replace("]","").concat("\n");
//            FileUtils.writeStringToFile(gradientsFile,gradientStr,true);
//
//            lktbTrainer.iterate();
//
//            SoftRegStump tree = ((SoftRegStump)lkTreeBoost.getRegressor(i,0));
//
//
//            int featurePicked = ((Sigmoid)tree.getGatingFunction()).getActiveFeatures().get(0);
//            double weight = ((Sigmoid)tree.getGatingFunction()).getWeights().get(featurePicked);
//            double bias = ((Sigmoid)tree.getGatingFunction()).getBias();
//            double threshold = -bias/weight;
//            double left = tree.getLeftOutput();
//            double right = tree.getRightOutput();
//            FileUtils.writeStringToFile(featuresFile,""+featurePicked+"\n",true);
//            FileUtils.writeStringToFile(thresFile,""+threshold+"\n",true);
//            FileUtils.writeStringToFile(leftFile,""+left+"\n",true);
//            FileUtils.writeStringToFile(rightFile,""+right+"\n",true);
//
//            double[] predctions = tree.predict(dataSet);
//
//            String predStr = Arrays.toString(predctions).replace("[","").replace("]","").concat("\n");
//            FileUtils.writeStringToFile(predFile,predStr,true);
//        }
////        System.out.println(lkTreeBoost);
//
//
//        List<Regressor> regressors = lkTreeBoost.getRegressors(0);
//        for (int i=0;i<regressors.size();i++){
//            System.out.println("iteration "+i);
//            Regressor regressor = regressors.get(i);
//            if (regressor instanceof RegressionTree){
//                System.out.println("hard tree");
//            }
//            if (regressor instanceof SoftRegStump){
//                SoftRegStump softRegStump = (SoftRegStump)regressor;
//                if (softRegStump.getLossType()== SoftRegStumpTrainer.LossType.SquaredLossOfExpectation){
//                    System.out.println("expectation tree");
//                }
//                if (softRegStump.getLossType()== SoftRegStumpTrainer.LossType.ExpectationOfSquaredLoss){
//                    System.out.println("probabilistic tree");
//                }
//            }
//        }
//
//        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
//                DataSetType.CLF_SPARSE, true);
//        System.out.println("test accuracy = "+Accuracy.accuracy(lkTreeBoost,testSet));
//    }
//}
