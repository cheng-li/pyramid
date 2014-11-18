package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.util.Sampling;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
import mltk.predictor.glm.ElasticNetLearner;
import mltk.predictor.glm.GLM;

import java.io.File;
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

    private static Config genHyperParams(){
        Config config = new Config();
        double lambda = Sampling.doubleLogUniform(0.001, 1000);
        double l1Ratio = Sampling.doubleUniform(0, 1);
        int iterations = Sampling.intUniform(50,200);
        config.setDouble("lambda",lambda);
        config.setDouble("l1Ratio",l1Ratio);
        config.setInt("iterations",iterations);
        return config;

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



        for (int run=0;run<config.getInt("numRuns");run++){
            Config hyperParams = genHyperParams();
            System.out.println("==============================");
            System.out.println("hyper parameters for the run:");
            System.out.println(hyperParams);
            double lambda = hyperParams.getDouble("lambda");
            double l1Ratio = hyperParams.getDouble("l1Ratio");
            int iterations = hyperParams.getInt("iterations");
            ElasticNetLearner learner = new ElasticNetLearner();
            learner.setTask(Learner.Task.CLASSIFICATION);
            learner.setMaxNumIters(iterations);
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
