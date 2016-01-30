package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMOptimizer;
import edu.neu.ccs.pyramid.util.Grid;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.io.FileUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;


/**
 * Created by Rainicy on 10/24/15.
 */
public class Exp211 {

    public static BMMOptimizer getOptimizer(Config config, BMMClassifier bmmClassifier, MultiLabelClfDataSet trainSet){
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,trainSet);

        optimizer.setMeanRegVariance(config.getDouble("lr.meanRegVariance"));
        optimizer.setMeanRegularization(config.getBoolean("lr.meanRegularization"));
        optimizer.setPriorVarianceMultiClass(config.getDouble("lr.multiClassVariance"));
        optimizer.setPriorVarianceBinary(config.getDouble("lr.binaryVariance"));
        optimizer.setNumIterationsBinary(config.getInt("boost.numIterationsBinary"));
        optimizer.setNumIterationsMultiClass(config.getInt("boost.numIterationsMultiClass"));
        optimizer.setShrinkageBinary(config.getDouble("boost.shrinkageBinary"));
        optimizer.setShrinkageMultiClass(config.getDouble("boost.shrinkageMultiClass"));
        optimizer.setNumLeavesBinary(config.getInt("boost.numLeavesBinary"));
        optimizer.setNumLeavesMultiClass(config.getInt("boost.numLeavesMultiClass"));

        return optimizer;
    }

    public static Pair<BMMClassifier,Integer> loadOldBMM(Config config) throws Exception{

        BMMClassifier bmmClassifier;
        int completedIterations = 0;
        String output = config.getString("output");
        String modelName = config.getString("modelName");
        File folder = Paths.get(output,modelName).toFile();
        File[] modeFiles = folder.listFiles((dir, name) -> name.startsWith("iter.") && (name.endsWith(".model")));
        File lastFile = null;
        int lastIter = -1;
        for (File file: modeFiles){
            String[] split = file.getName().split(Pattern.quote("."));
            int iter = Integer.parseInt(split[1]);
            if (iter>lastIter){
                lastIter = iter;
                lastFile = file;
                completedIterations = lastIter;
            }
        }
        bmmClassifier = BMMClassifier.deserialize(lastFile);
        System.out.println("bmm loaded, with "+completedIterations+ " iterations completed");
        bmmClassifier.setPredictMode(config.getString("predict.mode"));
        return new Pair<>(bmmClassifier,completedIterations);
    }

    public static Pair<BMMClassifier,Integer> loadNewBMM(Config config, MultiLabelClfDataSet trainSet) throws Exception{

        BMMClassifier bmmClassifier;
        int completedIterations = 0;

        bmmClassifier = BMMClassifier.getBuilder()
                .setNumClasses(trainSet.getNumClasses())
                .setNumFeatures(trainSet.getNumFeatures())
                .setNumClusters(config.getInt("mixture.numClusters"))
                .setMultiClassClassifierType(config.getString("mixture.multiClassClassifierType"))
                .setBinaryClassifierType(config.getString("mixture.binaryClassifierType"))
                .build();

        bmmClassifier.setPredictMode(config.getString("predict.mode"));
        bmmClassifier.setNumSample(config.getInt("predict.sampling.numSamples"));

        String allowEmpty = config.getString("predict.allowEmpty");
        switch (allowEmpty){
            case "true":
                bmmClassifier.setAllowEmpty(true);
                break;
            case "false":
                bmmClassifier.setAllowEmpty(false);
                break;
            case "auto":
                Set<MultiLabel> seen = DataSetUtil.gatherMultiLabels(trainSet).stream().collect(Collectors.toSet());
                MultiLabel empty = new MultiLabel();
                if (seen.contains(empty)){
                    bmmClassifier.setAllowEmpty(true);
                    System.out.println("training set contains empty labels, automatically set allow empty = true");
                } else {
                    bmmClassifier.setAllowEmpty(false);
                    System.out.println("training set does not contain empty labels, automatically set allow empty = false");
                }
                break;
            default:
                throw new IllegalArgumentException("unknown value for predict.allowEmpty");
        }

        if (config.getBoolean("train.initialize")) {
            System.out.println("start initialization with temperature "+config.getDouble("em.startTemperature"));
            BMMOptimizer optimizer = getOptimizer(config,bmmClassifier,trainSet);
            optimizer.setTemperature(config.getDouble("em.startTemperature"));
            BMMInitializer.initialize(bmmClassifier, trainSet, optimizer);
            System.out.println("finish initialization");
        }

        System.out.println("bmm loaded, with "+completedIterations+ " iterations completed");
        return new Pair<>(bmmClassifier,completedIterations);
    }



    public static Pair<BMMClassifier,Integer> loadBMM(Config config, MultiLabelClfDataSet trainSet) throws Exception{
        String mode = config.getString("train.warmStart");
        Pair<BMMClassifier,Integer> pair = null;
        switch (mode){
            case "true":
                pair = loadOldBMM(config);
                break;
            case "false":
                pair = loadNewBMM(config,trainSet);
                break;
            case "auto":
                String output = config.getString("output");
                String modelName = config.getString("modelName");
                File folder = Paths.get(output,modelName).toFile();
                File[] modeFiles = folder.listFiles((dir, name) -> name.startsWith("iter.") && (name.endsWith(".model")));
                if (modeFiles.length==0){
                    pair = loadNewBMM(config,trainSet);
                } else {
                    pair = loadOldBMM(config);
                }
                break;
            default:
                throw new IllegalArgumentException("unknown value for train.warmStart");
        }
        return pair;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);

        String matrixType = config.getString("input.matrixType");

        MultiLabelClfDataSet trainSet;
        MultiLabelClfDataSet testSet;

        switch (matrixType){
            case "sparse_random":
                 trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SPARSE, true);
                 testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SPARSE, true);
                break;
            case "sparse_sequential":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_SEQ_SPARSE, true);
                break;
            case "dense":
                trainSet= TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"),
                        DataSetType.ML_CLF_DENSE, true);
                testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"),
                        DataSetType.ML_CLF_DENSE, true);
                break;
            default:
                throw new IllegalArgumentException("unknown type");
        }


        int numIterations = config.getInt("em.numIterations");

        String output = config.getString("output");
        String modelName = config.getString("modelName");
        File path = Paths.get(output, modelName).toFile();
        path.mkdirs();

        Pair<BMMClassifier,Integer> pair = loadBMM(config,trainSet);
        BMMClassifier bmmClassifier = pair.getFirst();
        int completedIterations = pair.getSecond();

        BMMOptimizer optimizer = getOptimizer(config,bmmClassifier,trainSet);

        double startTemperature = config.getDouble("em.startTemperature");
        double endTemperature = config.getDouble("em.endTemperature");
        int numTemperatures = config.getInt("em.numTemperatures");
        List<Double> temperatures = Grid.uniformDecreasing(endTemperature,startTemperature,numTemperatures);

        int totalIter = completedIterations+1;
        for (double temperature: temperatures){
            System.out.println("------------------------------------------------");
            System.out.println("temperature = "+temperature);
            optimizer.setTemperature(temperature);

            for (int i=1;i<=numIterations;i++){
                System.out.print("iter : "+totalIter + "\t");
                optimizer.iterate();
                MultiLabel[] trainPredict;
                MultiLabel[] testPredict;
                trainPredict = bmmClassifier.predict(trainSet);
                testPredict = bmmClassifier.predict(testSet);
                System.out.print("objective: "+optimizer.getTerminator().getLastValue() + "\t");
                System.out.print("trainAcc : "+ Accuracy.accuracy(trainSet.getMultiLabels(),trainPredict)+ "\t");
                System.out.print("trainOver: "+ Overlap.overlap(trainSet.getMultiLabels(), trainPredict)+ "\t");
                System.out.print("testAcc  : "+ Accuracy.accuracy(testSet.getMultiLabels(),testPredict)+ "\t");
                System.out.println("testOver : "+ Overlap.overlap(testSet.getMultiLabels(), testPredict)+ "\t");
                if (config.getBoolean("saveModelForEachIter")) {
                    File serializeModel = new File(path,  "iter." + totalIter + ".model");
                    bmmClassifier.serialize(serializeModel);
                    double[][] gammas = optimizer.getGammas();
                    double[][] PIs = optimizer.getPIs();
                    BufferedWriter bw = new BufferedWriter(new FileWriter(new File(path, "iter."+totalIter+".gammas")));
                    BufferedWriter bw1 = new BufferedWriter(new FileWriter(new File(path, "iter."+totalIter+".PIs")));
                    for (int n=0; n<gammas.length; n++) {
                        for (int k=0; k<gammas[n].length; k++) {
                            bw.write(gammas[n][k] + "\t");
                            bw1.write(PIs[n][k] + "\t");
                        }
                        bw.write("\n");
                        bw1.write("\n");
                    }
                    bw.close();
                    bw1.close();
                }
                totalIter += 1;
            }
        }



        System.out.println("history = "+optimizer.getTerminator().getHistory());


        System.out.println("--------------------------------Results-----------------------------\n");
        System.out.println();
        System.out.print("trainAcc : " + Accuracy.accuracy(bmmClassifier, trainSet) + "\t");
        System.out.print("trainOver: "+ Overlap.overlap(bmmClassifier, trainSet)+ "\t");
        System.out.print("testAcc  : "+ Accuracy.accuracy(bmmClassifier,testSet)+ "\t");
        System.out.println("testOver : "+ Overlap.overlap(bmmClassifier, testSet)+ "\t");
        System.out.println();
        System.out.println();
//        System.out.println(bmmClassifier);

        if (config.getBoolean("saveModel")) {

            File serializeModel = new File(path, "model");
            bmmClassifier.serialize(serializeModel);
        }
    }
}
