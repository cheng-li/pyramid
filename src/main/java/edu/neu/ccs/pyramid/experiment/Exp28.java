//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.*;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//import edu.neu.ccs.pyramid.util.Pair;
//import edu.neu.ccs.pyramid.util.Sampling;
//import mltk.core.Instances;
//import mltk.predictor.Learner;
//import mltk.predictor.glm.GLM;
//import mltk.predictor.glm.RidgeLearner;
//
//import java.io.File;
//import java.util.stream.IntStream;
//
///**
// * ridge hyper parameter tuning
// * Created by chengli on 11/22/14.
// */
//public class Exp28 {
//    public static void main(String[] args) throws Exception{
//        if (args.length !=1){
//            throw new IllegalArgumentException("Please specify a properties file.");
//        }
//
//        Config config = new Config(args[0]);
//        System.out.println(config);
//        run(config);
//
//
//    }
//
//    private static Config genHyperParams(Config config){
//        Config hyperParams = new Config();
//        double lambda = Sampling.doubleLogUniform(config.getDoubles("lambda").get(0), config.getDoubles("lambda").get(1));
//        int iterations = Sampling.intUniform(config.getIntegers("iterations").get(0),config.getIntegers("iterations").get(1));
//        hyperParams.setDouble("lambda", lambda);
//        hyperParams.setInt("iterations", iterations);
//        return hyperParams;
//
//    }
//
//    private static void run(Config config) throws Exception{
//        Config bestParams;
//        double bestPerformance = Double.NEGATIVE_INFINITY;
//
//        for (int trial=0;trial<config.getInt("numTrials");trial++){
//            Config hyperParams = genHyperParams(config);
//            System.out.println("==============================");
//            System.out.println("hyper parameters for the trial:");
//            System.out.println(hyperParams);
//            double aveValidAcc= 0;
//            for (int validation=0;validation<config.getInt("numValidations");validation++){
//                System.out.println("validation "+validation);
//                Pair<ClfDataSet,ClfDataSet> dataSets = loadDataSets(config);
//
//                Instances trainSet = MLTKFormat.toInstances(dataSets.getFirst());
//
//                Instances validationSet = MLTKFormat.toInstances(dataSets.getSecond());
//
//
//
//                double lambda = hyperParams.getDouble("lambda");
//                int iterations = hyperParams.getInt("iterations");
//                RidgeLearner learner = new RidgeLearner();
//                learner.setTask(Learner.Task.CLASSIFICATION);
//                learner.setMaxNumIters(iterations);
//                learner.setVerbose(config.getBoolean("verbose"));
//                learner.setLambda(lambda);
//                GLM glm = learner.build(trainSet);
//                int[] predictions= IntStream.range(0, trainSet.size()).map(i-> glm.classify(trainSet.get(i)))
//                        .toArray();
//                int[] labels = IntStream.range(0,trainSet.size()).map(i-> (int)(trainSet.get(i).getTarget()))
//                        .toArray();
//                System.out.println("accuracy on the training set = "+ Accuracy.accuracy(labels, predictions));
//
//
//                int[] validationPredictions= IntStream.range(0,validationSet.size()).map(i -> glm.classify(validationSet.get(i)))
//                        .toArray();
//                int[] validationLabels = IntStream.range(0,validationSet.size()).map(i-> (int)(validationSet.get(i).getTarget()))
//                        .toArray();
//                double validationAcc = Accuracy.accuracy(validationLabels, validationPredictions);
//                System.out.println("accuracy on the validation set = "+validationAcc);
//                aveValidAcc += validationAcc;
//            }
//
//            aveValidAcc /= config.getInt("numValidations");
//
//            System.out.println("average accuracy on the validation set = "+ aveValidAcc);
//            if (aveValidAcc>bestPerformance){
//                bestPerformance = aveValidAcc;
//                bestParams = hyperParams;
//                System.out.println("**************************************");
//                System.out.println("best performance got so far: "+bestPerformance);
//                System.out.println("best hyper parameters got so far: ");
//                System.out.println(bestParams);
//                System.out.println("**************************************");
//            }
//        }
//    }
//
//    private static Pair<ClfDataSet, ClfDataSet> loadDataSets(Config config) throws Exception{
//        File trecFile = new File(config.getString("input.folder"),
//                config.getString("input.trainData"));
//        ClfDataSet clfDataSet = TRECFormat.loadClfDataSet(trecFile, DataSetType.CLF_SPARSE, false);
//        return DataSetUtil.splitToTrainValidation(clfDataSet, config.getDouble("trainPercentage"));
//
//    }
//}
