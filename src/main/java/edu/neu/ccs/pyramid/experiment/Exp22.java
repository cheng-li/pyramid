package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Sampling;
import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Learner;
import mltk.predictor.glm.ElasticNetLearner;
import mltk.predictor.glm.GLM;
import org.apache.commons.math3.distribution.LogNormalDistribution;

import java.io.File;
import java.util.stream.IntStream;

/**
 * elastic net hyper parameter tuning
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
        double lambda = Sampling.doubleLogUniform(0.001,1);
        double l1Ratio = Sampling.doubleUniform(0, 1);
        int iterations = Sampling.intUniform(50,200);
        config.setDouble("lambda",lambda);
        config.setDouble("l1Ratio",l1Ratio);
        config.setInt("iterations",iterations);
        return config;

    }

    private static void run(Config config) throws Exception{
        for (int run=0;run<config.getInt("numRuns");run++){
            Pair<ClfDataSet,ClfDataSet> dataSets = loadDataSets(config);

            Instances trainSet = MLTKFormat.toInstances(dataSets.getFirst());

            Instances validationSet = MLTKFormat.toInstances(dataSets.getSecond());

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
            learner.setVerbose(config.getBoolean("verbose"));
            learner.setLambda(lambda);
            learner.setL1Ratio(l1Ratio);
            GLM glm = learner.build(trainSet);
            int[] predictions= IntStream.range(0, trainSet.size()).map(i-> glm.classify(trainSet.get(i)))
                    .toArray();
            int[] labels = IntStream.range(0,trainSet.size()).map(i-> (int)(trainSet.get(i).getTarget()))
                    .toArray();
            System.out.println("accuracy on the training set = "+ Accuracy.accuracy(labels, predictions));


            int[] validationPredictions= IntStream.range(0,validationSet.size()).map(i -> glm.classify(validationSet.get(i)))
                    .toArray();
            int[] validationLabels = IntStream.range(0,validationSet.size()).map(i-> (int)(validationSet.get(i).getTarget()))
                    .toArray();
            System.out.println("accuracy on the validation set = "+ Accuracy.accuracy(validationLabels,validationPredictions));
        }
    }
    
    private static Pair<ClfDataSet, ClfDataSet> loadDataSets(Config config) throws Exception{
        File trecFile = new File(config.getString("input.folder"),
                config.getString("input.trainData"));
        ClfDataSet clfDataSet = TRECFormat.loadClfDataSet(trecFile, DataSetType.CLF_SPARSE,false);
        return DataSetUtil.splitToTrainValidation(clfDataSet,config.getDouble("trainPercentage"));
        
    }

}
