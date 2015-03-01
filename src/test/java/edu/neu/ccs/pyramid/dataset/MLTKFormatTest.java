//package edu.neu.ccs.pyramid.dataset;
//
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//import mltk.core.Instances;
//import mltk.core.io.InstancesReader;
//import mltk.predictor.Learner;
//import mltk.predictor.glm.GLM;
//import mltk.predictor.glm.LassoLearner;
//
//import java.io.File;
//import java.util.stream.IntStream;
//
//import static org.junit.Assert.*;
//
//public class MLTKFormatTest {
//    private static final Config config = new Config("configs/local.config");
//    private static final String DATASETS = config.getString("input.datasets");
//    private static final String TMP = config.getString("output.tmp");
//
//    public static void main(String[] args) throws Exception{
//        test1();
//    }
//
//    private static void test1() throws Exception{
//        GLM glm = spamTrain();
//        spamTest(glm);
//    }
//
//    private static GLM spamTrain() throws Exception{
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
//                DataSetType.CLF_DENSE,true);
//        System.out.println(dataSet.getRow(0));
//        Instances trainSet = MLTKFormat.toInstances(dataSet);
//        System.out.println(trainSet.get(0));
//        System.out.println(trainSet.dimension());
//        System.out.println(trainSet.getAttributes());
//        System.out.println(trainSet.getTargetAttribute());
//        LassoLearner learner = new LassoLearner();
//        learner.setTask(Learner.Task.CLASSIFICATION);
//        learner.setMaxNumIters(100);
//        learner.setVerbose(false);
//        learner.setLambda(0.1);
//        GLM glm = learner.build(trainSet);
//        int[] predictions= IntStream.range(0, trainSet.size()).map(i-> glm.classify(trainSet.get(i)))
//                .toArray();
//        int[] labels = IntStream.range(0,trainSet.size()).map(i-> (int)(trainSet.get(i).getTarget()))
//                .toArray();
//        System.out.println("accuracy on the training set = "+ Accuracy.accuracy(labels, predictions));
//        return glm;
//    }
//
//    private static void spamTest(GLM glm) throws Exception{
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
//                DataSetType.CLF_DENSE, true);
//        Instances testSet = MLTKFormat.toInstances(dataSet);
//        int[] predictions= IntStream.range(0,testSet.size()).map(i-> glm.classify(testSet.get(i)))
//                .toArray();
//        int[] labels = IntStream.range(0,testSet.size()).map(i-> (int)(testSet.get(i).getTarget()))
//                .toArray();
//        System.out.println("accuracy on the test set = "+ Accuracy.accuracy(labels,predictions));
//    }
//
//    private static void test2() throws Exception{
//        String matrixFile = new File(new File(DATASETS,"/spam/trec_data/train.trec"), "feature_matrix.txt").getAbsolutePath();
//        Instances trainSet = InstancesReader.read(null, matrixFile);
//        System.out.println(trainSet.getAttributes());
//        System.out.println(trainSet.getTargetAttribute());
//
//    }
//
//}