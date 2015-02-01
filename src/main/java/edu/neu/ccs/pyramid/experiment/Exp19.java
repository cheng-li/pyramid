package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBConfig;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBInspector;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * single label gradient boosting without probabilistic voting
 * Created by chengli on 11/8/14.
 */
public class Exp19 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        if (config.getBoolean("train")){
            train(config);
        }

        if (config.getBoolean("verify")){
            verify(config);
        }

        if (config.getBoolean("test")){
            test(config);
        }


    }

    static void train(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        int numIterations = config.getInt("train.numIterations");
        int numLeaves = config.getInt("train.numLeaves");
        double learningRate = config.getDouble("train.learningRate");
        int minDataPerLeaf = config.getInt("train.minDataPerLeaf");
        String modelName = config.getString("archive.model");
        double featureSamplingRate = config.getDouble("train.featureSamplingRate");
        double dataSamplingRate = config.getDouble("train.dataSamplingRate");
        boolean overwriteModels = config.getBoolean("train.overwriteModels");



        //re-sample
        ClfDataSet dataSet = loadTrainData(config);
        int numClasses = dataSet.getNumClasses();

        System.out.println("training model ");
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .learningRate(learningRate).minDataPerLeaf(minDataPerLeaf)
                .numLeaves(numLeaves).dataSamplingRate(dataSamplingRate)
                .featureSamplingRate(featureSamplingRate).build();
        LKTreeBoost lkTreeBoost = new LKTreeBoost(numClasses);
        lkTreeBoost.setPriorProbs(dataSet);
        lkTreeBoost.setTrainConfig(trainConfig);
        for (int i=0;i<numIterations;i++){

            System.out.println("iteration "+i);

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
        File serializedModel =  new File(archive,modelName);
        if (!overwriteModels && serializedModel.exists()){
            throw new RuntimeException(serializedModel.getAbsolutePath()+"already exists");
        }

        lkTreeBoost.serialize(serializedModel);
        System.out.println(stopWatch);
        System.out.println("accuracy on training set = "+ Accuracy.accuracy(lkTreeBoost,
                dataSet));

    }



    static void verify(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();

        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_SPARSE,
                    true);
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_DENSE,
                    true);
        }
        File serializedModel =  new File(archive,modelName);
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
        double acc = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println("accuracy on training set = "+acc);
        System.out.println("MRR on training set = "+
                MRR.mrr(lkTreeBoost, dataSet));
        int top = config.getInt("verify.appearanceAtTop.top");
        System.out.println("rate of label appearance at top "+top+" on training set = "+
                + AppearanceAtTop.rate(lkTreeBoost, dataSet, top));
        if (config.getBoolean("verify.confusionMatrix")){
            showConfusionMatrix(lkTreeBoost,dataSet);
        }


        if (config.getBoolean("verify.topFeatures")){
            getTopFeatures(config,dataSet.getNumClasses());
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
                    true);
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_DENSE,
                    true);
        }


        return dataSet;
    }


    static void test(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");
        String testFile = new File(config.getString("input.folder"),
                config.getString("input.testData")).getAbsolutePath();

        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(testFile), DataSetType.CLF_SPARSE,
                    true);
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(testFile), DataSetType.CLF_DENSE,
                    true);
        }
        File serializedModel =  new File(archive,modelName);
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
        double acc = Accuracy.accuracy(lkTreeBoost,dataSet);
        System.out.println("accuracy on test set = "+acc);
        System.out.println("MRR on test set = "+
                MRR.mrr(lkTreeBoost, dataSet));
        int top = config.getInt("test.appearanceAtTop.top");
        System.out.println("rate of label appearance at top "+top+" on test set = "+
                + AppearanceAtTop.rate(lkTreeBoost, dataSet, top));
        if (config.getBoolean("test.confusionMatrix")){
            showConfusionMatrix(lkTreeBoost,dataSet);
        }

        LabelTranslator labelTranslator = dataSet.getSetting().getLabelTranslator();

        if (config.getBoolean("test.mistakes")){
            System.out.println("============analyzing mistakes===========");
            int limit = config.getInt("test.mistakes.limit");
            for (int i=0;i<dataSet.getNumDataPoints();i++){
                Vector row = dataSet.getRow(i);
                int label = dataSet.getLabels()[i];
                int prediction = lkTreeBoost.predict(row);
                if (label!=prediction){
                    System.out.println("data point "+i+", extid = "+dataSet.getDataPointSetting(i).getExtId());
                    System.out.println(LKTBInspector.analyzeMistake(lkTreeBoost,row,label,prediction,labelTranslator,limit));
                }
            }
        }

    }



    static void getTopFeatures(Config config, int numClasses) throws Exception{
        LabelTranslator labelTranslator = loadLabelTranslator(config);
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");


        File serializedModel =  new File(archive,modelName);
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);

        System.out.println("==========top features==========");
        for (int k=0;k<numClasses;k++){
            List<String> features = LKTBInspector.topFeatureNames(lkTreeBoost, k);
            System.out.println("top features for class "+k+"("+labelTranslator.toExtLabel(k)+"):");
            System.out.println(features);
        }


    }



    static void showTrees(Config config) throws Exception{
        String archive = config.getString("archive.folder");
        String modelName = config.getString("archive.model");
        File serializedModel =  new File(archive,modelName);
        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(serializedModel);
        System.out.println(lkTreeBoost);

    }

    static void showConfusionMatrix(Classifier classifier,
                                    ClfDataSet dataSet){
        ConfusionMatrix confusionMatrix = new ConfusionMatrix(classifier,dataSet);
        System.out.println("==========confusion matrix==========");
        System.out.println(confusionMatrix.printWithExtLabels());
        System.out.println("==========micro averaged measures==========");
        MicroAveragedMeasures microAveragedMeasures = new MicroAveragedMeasures(confusionMatrix);
        System.out.println(microAveragedMeasures);
        System.out.println("==========macro averaged measures==========");
        MacroAveragedMeasures macroAveragedMeasures = new MacroAveragedMeasures(confusionMatrix);
        System.out.println(macroAveragedMeasures);
    }

    static LabelTranslator loadLabelTranslator(Config config) throws Exception{
        String trainFile = new File(config.getString("input.folder"),
                config.getString("input.trainData")).getAbsolutePath();
        ClfDataSet dataSet;
        if (config.getBoolean("featureMatrix.sparse")){
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_SPARSE,
                    true);
        } else {
            dataSet= TRECFormat.loadClfDataSet(new File(trainFile), DataSetType.CLF_DENSE,
                    true);
        }

        return dataSet.getSetting().getLabelTranslator();
    }
}
