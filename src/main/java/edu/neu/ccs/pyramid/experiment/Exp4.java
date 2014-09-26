package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.ProbabilityVoting;
import edu.neu.ccs.pyramid.classification.Voting;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBConfig;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBInspector;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.AppearanceAtTop;
import edu.neu.ccs.pyramid.eval.ConfusionMatrix;
import edu.neu.ccs.pyramid.eval.MRR;
import org.apache.commons.lang3.time.StopWatch;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * feature sampling gradient boosting
* Created by chengli on 9/9/14.
*/
public class Exp4 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        if (config.getBoolean("train")){
            train(config);
        }
        if (config.getBoolean("test")){
            test(config);
        }
        if (config.getBoolean("verify")){
            verify(config);
        }

    }

    static void train(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        int numIterations = config.getInt("train.numIterations");
        int numClasses = config.getInt("numClasses");
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int minDataPerLeaf = config.getInt("train.minDataPerLeaf");
        String modelName = config.getString("archive.model");
        double featureSamplingRate = config.getDouble("train.featureSamplingRate");
        double dataSamplingRate = config.getDouble("train.dataSamplingRate");
        int firstModel = config.getInt("train.firstModel");
        int lastModel = config.getInt("train.lastModel");
        boolean overwriteModels = config.getBoolean("train.overwriteModels");

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        for (int modelIndex=firstModel;modelIndex<=lastModel;modelIndex++){
            //re-sample
            ClfDataSet dataSet = loadTrainData(config);

            System.out.println("training model "+modelIndex);

            LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet,numClasses)
                    .learningRate(learningRate).minDataPerLeaf(minDataPerLeaf)
                    .numLeaves(numLeaves).dataSamplingRate(dataSamplingRate)
                    .featureSamplingRate(featureSamplingRate).build();
            LKTreeBoost lkTreeBoost = new LKTreeBoost(numClasses);
            lkTreeBoost.setPriorProbs(dataSet);
            lkTreeBoost.setTrainConfig(trainConfig);
            for (int i=0;i<numIterations;i++){

//                System.out.println("iteration "+i);

                lkTreeBoost.boostOneRound();
                //debug
//                double[] gradient = lkTreeBoost.getGradient(0);
//                List<String> extIds = IntStream.range(0, gradient.length)
//                        .mapToObj(a -> new IntDoublePair(a,gradient[a]))
//                        .filter(intDoublePair -> intDoublePair.getValue()>0)
//                        .sorted(Comparator.comparing(IntDoublePair::getValue).reversed())
//                        .mapToInt(IntDoublePair::getIndex).mapToObj(idTranslator::toIndexId)
//                        .collect(Collectors.toList());
//                System.out.println("high gradient docs: ");
//                System.out.println(extIds);
//                double[] gradientArray = IntStream.range(0, gradient.length)
//                        .mapToObj(a -> new IntDoublePair(a,gradient[a]))
//                        .filter(intDoublePair -> intDoublePair.getValue()>0)
//                        .sorted(Comparator.comparing(IntDoublePair::getValue).reversed())
//                        .mapToDouble(IntDoublePair::getValue).toArray();
//                System.out.println(Arrays.toString(gradientArray));
            }
            File serializedModel =  new File(archive,modelName+"_"+modelIndex);
            if (!overwriteModels && serializedModel.exists()){
                throw new RuntimeException(serializedModel.getAbsolutePath()+"already exists");
            }

            LKTreeBoost.serialize(lkTreeBoost,serializedModel);
            System.out.println(stopWatch);
            System.out.println("accuracy on training set = "+ Accuracy.accuracy(lkTreeBoost,
                    dataSet));
        }
    }

    static void test(Config config) throws Exception{

        if (config.getBoolean("test.model")){
            modelTest(config);
        }
        if (config.getBoolean("test.voting")){
            votingTest(config);
        }
        if (config.getBoolean("test.probVoting")){
            probVotingTest(config);
        }
    }

    static void verify(Config config) throws Exception{
        if (config.getBoolean("verify.model")){
            modelVerify(config);
        }
        if (config.getBoolean("verify.voting")){
            votingVerify(config);
        }
        if (config.getBoolean("verify.probVoting")){
            probVotingVerify(config);
        }


        if (config.getBoolean("verify.topFeatures")){
            getTopFeatures(config);
        }

        if (config.getBoolean("verify.showTrees")){
            showTrees(config);
        }

    }

    static ClfDataSet loadTrainData(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        ClfDataSet dataSet;

        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_SPARSE,
                    config.getBoolean("input.loadSettings"));
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_DENSE,
                    config.getBoolean("input.loadSettings"));
        }

        ClfDataSet trainDataSet;
        if (config.getBoolean("train.bootstrap")){
            trainDataSet = DataSetUtil.bootstrap(dataSet);
        } else {
            trainDataSet = dataSet;
        }
        return trainDataSet;
    }


    static void modelTest(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");
        String testFile = new File(config.getString("input.folder"),
                config.getString("input.testData")).getAbsolutePath();
        int firstModel = config.getInt("test.firstModel");
        int lastModel = config.getInt("test.lastModel");

        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(testFile), DataSetType.CLF_SPARSE,
                    config.getBoolean("input.loadSettings"));
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(testFile), DataSetType.CLF_DENSE,
                    config.getBoolean("input.loadSettings"));
        }
        double aveAcc = 0;
        for (int modelIndex =firstModel;modelIndex<=lastModel;modelIndex++){
            File serializedModel =  new File(archive,modelName+"_"+modelIndex);
            LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
            double acc = Accuracy.accuracy(lkTreeBoost,dataSet);
            System.out.println("model "+modelIndex+" accuracy on test set = "+acc);
            aveAcc += acc;
        }
        aveAcc /= (lastModel-firstModel+1);
        System.out.println("average accuracy of models "+firstModel+" - "+lastModel+" on test set ="+aveAcc);
    }

    static void votingTest(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");
        String testFile = new File(config.getString("input.folder"),
                config.getString("input.testData")).getAbsolutePath();
        int numClasses = config.getInt("numClasses");
        int firstModel = config.getInt("test.firstModel");
        int lastModel = config.getInt("test.lastModel");
        boolean checkIncreasingRate = config.getBoolean("test.voting.checkIncreasingRate");

        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(testFile), DataSetType.CLF_SPARSE,
                    config.getBoolean("input.loadSettings"));
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(testFile), DataSetType.CLF_DENSE,
                    config.getBoolean("input.loadSettings"));
        }

        Voting voting = new Voting(numClasses);
        for (int modelIndex =firstModel;modelIndex<=lastModel;modelIndex++){
            File serializedModel =  new File(archive,modelName+"_"+modelIndex);
            LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
            voting.add(lkTreeBoost);
            if (checkIncreasingRate){
                System.out.println("voting of models ("+firstModel+" - "+modelIndex +") accuracy on test set = "+
                        Accuracy.accuracy(voting,dataSet));
            }
        }
        System.out.println("voting of models ("+firstModel+" - "+lastModel +") accuracy on test set = "+
                Accuracy.accuracy(voting,dataSet));

    }

    static void probVotingTest(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");
        String testFile = new File(config.getString("input.folder"),
                config.getString("input.testData")).getAbsolutePath();
        int numClasses = config.getInt("numClasses");
        int firstModel = config.getInt("test.firstModel");
        int lastModel = config.getInt("test.lastModel");
        boolean checkIncreasingRate = config.getBoolean("test.probVoting.checkIncreasingRate");

        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(testFile), DataSetType.CLF_SPARSE,
                    config.getBoolean("input.loadSettings"));
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(testFile), DataSetType.CLF_DENSE,
                    config.getBoolean("input.loadSettings"));
        }
        ProbabilityVoting probabilityVoting = new ProbabilityVoting(numClasses);

        for (int modelIndex =firstModel;modelIndex<=lastModel;modelIndex++){
            File serializedModel =  new File(archive,modelName+"_"+modelIndex);
            LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
            probabilityVoting.add(lkTreeBoost);
            if (checkIncreasingRate){
                System.out.println("probability voting of models ("+firstModel+" - "+modelIndex +") accuracy on test set = "+
                        Accuracy.accuracy(probabilityVoting,dataSet));
            }
        }
        System.out.println("probability voting of models ("+firstModel+" - "+lastModel +") accuracy on test set = "+
                Accuracy.accuracy(probabilityVoting,dataSet));
        System.out.println("probability voting of models ("+firstModel+" - "+lastModel +") MRR on test set = "+
                MRR.mrr(probabilityVoting, dataSet));
        int top = config.getInt("test.appearanceAtTop.top");
        System.out.println("probability voting of models ("+firstModel+" - "+lastModel +") rate of label appearance at top "+top+" on test set = "+
                + AppearanceAtTop.rate(probabilityVoting, dataSet, top));

        if (config.getBoolean("test.probVoting.confusionMatrix")){
            showConfusionMatrix(numClasses,probabilityVoting,dataSet);
        }

    }



    static void modelVerify(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        int firstModel = config.getInt("verify.firstModel");
        int lastModel = config.getInt("verify.lastModel");

        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_SPARSE,
                    config.getBoolean("input.loadSettings"));
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_DENSE,
                    config.getBoolean("input.loadSettings"));
        }
        double aveAcc = 0;
        for (int modelIndex =firstModel;modelIndex<=lastModel;modelIndex++){
            File serializedModel =  new File(archive,modelName+"_"+modelIndex);
            LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
            double acc = Accuracy.accuracy(lkTreeBoost,dataSet);
            System.out.println("model "+modelIndex+" accuracy on training set = "+acc);
            aveAcc += acc;
        }
        aveAcc /= (lastModel-firstModel+1);
        System.out.println("average accuracy of models "+firstModel+" - "+lastModel+" on training set ="+aveAcc);
    }

    static void votingVerify(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        int numClasses = config.getInt("numClasses");
        int firstModel = config.getInt("verify.firstModel");
        int lastModel = config.getInt("verify.lastModel");
        boolean checkIncreasingRate = config.getBoolean("verify.voting.checkIncreasingRate");

        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_SPARSE,
                    config.getBoolean("input.loadSettings"));
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_DENSE,
                    config.getBoolean("input.loadSettings"));
        }

        Voting voting = new Voting(numClasses);
        for (int modelIndex =firstModel;modelIndex<=lastModel;modelIndex++){
            File serializedModel =  new File(archive,modelName+"_"+modelIndex);
            LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
            voting.add(lkTreeBoost);
            if (checkIncreasingRate){
                System.out.println("voting of models ("+firstModel+" - "+modelIndex +") accuracy on training set = "+
                        Accuracy.accuracy(voting,dataSet));
            }
        }
        System.out.println("voting of models ("+firstModel+" - "+lastModel +") accuracy on training set = "+
                Accuracy.accuracy(voting,dataSet));
    }

    static void probVotingVerify(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        int numClasses = config.getInt("numClasses");
        int firstModel = config.getInt("verify.firstModel");
        int lastModel = config.getInt("verify.lastModel");
        boolean checkIncreasingRate = config.getBoolean("verify.probVoting.checkIncreasingRate");

        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_SPARSE,
                    config.getBoolean("input.loadSettings"));
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_DENSE,
                    config.getBoolean("input.loadSettings"));
        }
        ProbabilityVoting probabilityVoting = new ProbabilityVoting(numClasses);

        for (int modelIndex =firstModel;modelIndex<=lastModel;modelIndex++){
            File serializedModel =  new File(archive,modelName+"_"+modelIndex);
            LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
            probabilityVoting.add(lkTreeBoost);
            if (checkIncreasingRate){
                System.out.println("probability voting of models ("+firstModel+" - "+modelIndex +") accuracy on training set = "+
                        Accuracy.accuracy(probabilityVoting,dataSet));
            }
        }
        System.out.println("probability voting of models ("+firstModel+" - "+lastModel +") accuracy on training set = "+
                Accuracy.accuracy(probabilityVoting,dataSet));
        System.out.println("probability voting of models ("+firstModel+" - "+lastModel +") MRR on training set = "+
                MRR.mrr(probabilityVoting, dataSet));
        int top = config.getInt("verify.appearanceAtTop.top");
        System.out.println("probability voting of models ("+firstModel+" - "+lastModel +") rate of label appearance at top "+top+" on training set = "+
                + AppearanceAtTop.rate(probabilityVoting, dataSet, top));
    }




    //todo get overall top features
    static void getTopFeatures(Config config) throws Exception{
        Map<Integer,String> labelMap = loadLabelMap(config);
        String archive = config.getString("archive.folder");
        int firstModel = config.getInt("verify.firstModel");
        int lastModel = config.getInt("verify.lastModel");
        String modelName = config.getString("archive.model");
        List<LKTreeBoost> lkTreeBoosts = new ArrayList<>();
        for (int modelIndex =firstModel;modelIndex<=lastModel;modelIndex++) {
            File serializedModel =  new File(archive,modelName+"_"+modelIndex);
            LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
            lkTreeBoosts.add(lkTreeBoost);
            if (config.getBoolean("verify.topFeatures.model")){
                System.out.println("model "+modelIndex);
                System.out.println("==========top features==========");
                for (int k=0;k<config.getInt("numClasses");k++){
                    List<String> features = LKTBInspector.topFeatureNames(lkTreeBoost, k);
                    System.out.println("top features for class "+k+"("+labelMap.get(k)+"):");
                    System.out.println(features);
                }
            }
        }
        if (config.getBoolean("verify.topFeatures.overall")){
            System.out.println("overall top features among all models:");
            for (int k=0;k<config.getInt("numClasses");k++){
                List<String> features = LKTBInspector.topFeatureNames(lkTreeBoosts, k);
                System.out.println("top features for class "+k+"("+labelMap.get(k)+"):");
                System.out.println(features);
                if (config.getBoolean("verify.topFeatures.writeToFiles")){
                    File featureFolder = new File(archive,"top_features");
                    if (!featureFolder.exists()){
                        featureFolder.mkdirs();
                    }
                    File featureFile = new File(featureFolder,""+k);
                    try (BufferedWriter bw = new BufferedWriter(new FileWriter(featureFile))
                        ){
                        bw.write(labelMap.get(k));
                        bw.newLine();
                      for (String feature: features){
                          bw.write(feature);
                          bw.newLine();
                        }
                    }
                }
            }
        }
    }



    static void showTrees(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        int firstModel = config.getInt("verify.firstModel");
        int lastModel = config.getInt("verify.lastModel");
        String modelName = config.getString("archive.model");
        for (int modelIndex =firstModel;modelIndex<=lastModel;modelIndex++) {
            File serializedModel =  new File(archive,modelName+"_"+modelIndex);
            LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
            System.out.println("model "+modelIndex);
            System.out.println(lkTreeBoost);
        }
    }

    static void showConfusionMatrix(int numClasses, Classifier classifier,
                                    ClfDataSet dataSet){
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(numClasses,classifier,dataSet);
        System.out.println("==========confusion matrix==========");
        System.out.println(confusionMatrix.printWithExtLabels());
    }

    static Map<Integer,String> loadLabelMap(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_SPARSE,
                    config.getBoolean("input.loadSettings"));
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_DENSE,
                    config.getBoolean("input.loadSettings"));
        }

        return dataSet.getSetting().getLabelMap();
    }
}
