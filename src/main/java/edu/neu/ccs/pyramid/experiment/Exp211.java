package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMInitializer;
import edu.neu.ccs.pyramid.multilabel_classification.bmm_variant.BMMOptimizer;
import org.apache.commons.io.FileUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Paths;


/**
 * Created by Rainicy on 10/24/15.
 */
public class Exp211 {

    private static BMMOptimizer getOptimizer(Config config, BMMClassifier bmmClassifier, MultiLabelClfDataSet trainSet){
        BMMOptimizer optimizer = new BMMOptimizer(bmmClassifier,trainSet);

        optimizer.setInverseTemperature(config.getDouble("em.inverseTemperature"));
        optimizer.setMeanRegVariance(config.getDouble("lr.meanRegVariance"));
        optimizer.setMeanRegularization(config.getBoolean("lr.meanRegularization"));
        optimizer.setPriorVarianceMultiClass(config.getDouble("lr.multiClassVariance"));
        optimizer.setPriorVarianceBinary(config.getDouble("lr.binaryVariance"));
        optimizer.setNumIterationsBinary(config.getInt("boost.numIterationsBinary"));
        optimizer.setNumIterationsMultiClass(config.getInt("boost.numIterationsMultiClass"));
        optimizer.setShrinkageBinary(config.getDouble("boost.shrinkageBinary"));
        optimizer.setShrinkageMultiClass(config.getDouble("boost.shrinkageMultiClass"));

        return optimizer;
    }

    public static BMMClassifier loadBMM(Config config, MultiLabelClfDataSet trainSet, MultiLabelClfDataSet testSet) throws Exception{
        int numClusters = config.getInt("mixture.numClusters");

        String output = config.getString("output");
        String modelName = config.getString("modelName");


        BMMClassifier bmmClassifier;
        if (config.getBoolean("train.warmStart")) {
            bmmClassifier = BMMClassifier.deserialize(new File(output, modelName));
        } else {
            bmmClassifier = BMMClassifier.getBuilder()
                    .setNumClasses(trainSet.getNumClasses())
                    .setNumFeatures(trainSet.getNumFeatures())
                    .setNumClusters(numClusters)
                    .setMultiClassClassifierType("mixture.multiClassClassifierType")
                    .setBinaryClassifierType("mixture.binaryClassifierType")
                    .build();

            bmmClassifier.setPredictMode(config.getString("predict.mode"));
            bmmClassifier.setNumSample(config.getInt("predict.sampling.numSamples"));
            bmmClassifier.setAllowEmpty(config.getBoolean("predict.allowEmpty"));

            MultiLabel[] trainPredict;
            MultiLabel[] testPredict;


            if (config.getBoolean("train.initialize")) {
                System.out.println("start initialization");
                BMMOptimizer optimizer = getOptimizer(config,bmmClassifier,trainSet);
                BMMInitializer.initialize(bmmClassifier, trainSet, optimizer);
                System.out.println("finish initialization");
            }
            trainPredict = bmmClassifier.predict(trainSet);
            testPredict = bmmClassifier.predict(testSet);

            System.out.print("trainAcc : " + Accuracy.accuracy(trainSet.getMultiLabels(), trainPredict) + "\t");
            System.out.print("trainOver: " + Overlap.overlap(trainSet.getMultiLabels(), trainPredict) + "\t");
            System.out.print("testACC  : " + Accuracy.accuracy(testSet.getMultiLabels(), testPredict) + "\t");
            System.out.println("testOver : " + Overlap.overlap(testSet.getMultiLabels(), testPredict) + "\t");

        }

        return bmmClassifier;
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
            default:
                throw new IllegalArgumentException("unknown type");
        }


        int numIterations = config.getInt("em.numIterations");

        String output = config.getString("output");
        String modelName = config.getString("modelName");
        File path = Paths.get(output,modelName).toFile();
        path.mkdirs();
        FileUtils.cleanDirectory(path);

        BMMClassifier bmmClassifier = loadBMM(config,trainSet,testSet);

        BMMOptimizer optimizer = getOptimizer(config,bmmClassifier,trainSet);

        for (int i=1;i<=numIterations;i++){
            System.out.print("iter : "+i + "\t");
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
                File serializeModel = new File(path,  "iter." + i + ".model");
                bmmClassifier.serialize(serializeModel);
                double[][] gammas = optimizer.getGammas();
                double[][] PIs = optimizer.getPIs();
                BufferedWriter bw = new BufferedWriter(new FileWriter(new File(path, "iter."+i+".gammas")));
                BufferedWriter bw1 = new BufferedWriter(new FileWriter(new File(path, "iter."+i+".PIs")));
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
