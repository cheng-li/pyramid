package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.eval.Accuracy;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
import mltk.predictor.glm.ElasticNetLearner;
import mltk.predictor.glm.GLM;
import mltk.predictor.glm.LassoLearner;

import java.io.File;
import java.util.List;
import java.util.stream.IntStream;

/**
 * elastic net
 * Created by chengli on 11/12/14.
 */
public class Exp22 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        run(config);


    }

    private static void run(Config config) throws Exception{
        File trecFile = new File(config.getString("input.folder"),
                config.getString("input.trainData"));
        String matrixFile = new File(trecFile, "feature_matrix.txt").getAbsolutePath();
        Instances trainSet = InstancesReader.read(null, matrixFile);

        File testFile = new File(config.getString("input.folder"),
                config.getString("input.testData"));
        String testMatrixFile = new File(testFile, "feature_matrix.txt").getAbsolutePath();
        Instances testSet = InstancesReader.read(null,testMatrixFile);

        List<Double> lambdas = config.getDoubles("train.lambda");
        List<Double> l1Ratios = config.getDoubles("train.l1Ratio");

        for (double lambda: lambdas){
            for (double l1Ratio: l1Ratios){
                System.out.println("lambda = "+lambda);
                System.out.println("l1Ratio = "+l1Ratio);
                ElasticNetLearner learner = new ElasticNetLearner();
                learner.setTask(Learner.Task.CLASSIFICATION);
                learner.setMaxNumIters(config.getInt("train.numIterations"));
                learner.setVerbose(false);
                learner.setLambda(lambda);
                learner.setL1Ratio(l1Ratio);
                GLM glm = learner.build(trainSet);
                int[] predictions= IntStream.range(0, trainSet.size()).map(i-> glm.classify(trainSet.get(i)))
                        .toArray();
                int[] labels = IntStream.range(0,trainSet.size()).map(i-> (int)(trainSet.get(i).getTarget()))
                        .toArray();
                System.out.println("accuracy on the training set = "+ Accuracy.accuracy(labels, predictions));


                int[] testPredictions= IntStream.range(0,testSet.size()).map(i-> glm.classify(testSet.get(i)))
                        .toArray();
                int[] testLabels = IntStream.range(0,testSet.size()).map(i-> (int)(testSet.get(i).getTarget()))
                        .toArray();
                System.out.println("accuracy on the test set = "+ Accuracy.accuracy(testLabels,testPredictions));
            }
        }


    }

}
