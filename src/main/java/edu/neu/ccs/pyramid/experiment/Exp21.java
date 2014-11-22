package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.eval.Accuracy;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
import mltk.predictor.glm.GLM;
import mltk.predictor.glm.LassoLearner;
import mltk.predictor.glm.RidgeLearner;

import java.io.File;
import java.util.stream.IntStream;

/**
 * ridge
 * todo
 * Created by chengli on 11/12/14.
 */
public class Exp21 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        GLM glm = train(config);
        test(config,glm);

    }

    private static GLM train(Config config) throws Exception{
        File trecFile = new File(config.getString("input.folder"),
                config.getString("input.trainData"));
        String matrixFile = new File(trecFile, "feature_matrix.txt").getAbsolutePath();
        Instances trainSet = InstancesReader.read(null, matrixFile);
        RidgeLearner learner = new RidgeLearner();
        learner.setTask(Learner.Task.CLASSIFICATION);
        learner.setMaxNumIters(config.getInt("train.numIterations"));
        learner.setVerbose(false);
        learner.setLambda(config.getDouble("train.lambda"));
        GLM glm = learner.build(trainSet);
        int[] predictions= IntStream.range(0, trainSet.size()).map(i-> glm.classify(trainSet.get(i)))
                .toArray();
        int[] labels = IntStream.range(0,trainSet.size()).map(i-> (int)(trainSet.get(i).getTarget()))
                .toArray();
        System.out.println("accuracy on the training set = "+ Accuracy.accuracy(labels, predictions));
        return glm;
    }

    private static void test(Config config, GLM glm) throws Exception{
        File trecFile = new File(config.getString("input.folder"),
                config.getString("input.testData"));
        String matrixFile = new File(trecFile, "feature_matrix.txt").getAbsolutePath();
        Instances testSet = InstancesReader.read(null,matrixFile);
        int[] predictions= IntStream.range(0,testSet.size()).map(i-> glm.classify(testSet.get(i)))
                .toArray();
        int[] labels = IntStream.range(0,testSet.size()).map(i-> (int)(testSet.get(i).getTarget()))
                .toArray();
        System.out.println("accuracy on the test set = "+ Accuracy.accuracy(labels,predictions));
    }
}
