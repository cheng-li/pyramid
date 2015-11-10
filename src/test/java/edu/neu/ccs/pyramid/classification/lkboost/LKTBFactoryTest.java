//package edu.neu.ccs.pyramid.classification.lkboost;
//
//import edu.neu.ccs.pyramid.classification.Classifier;
//import edu.neu.ccs.pyramid.classification.TrainConfig;
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.ClfDataSet;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//import edu.neu.ccs.pyramid.eval.ConfusionMatrix;
//
//import java.io.File;
//
//public class LKTBFactoryTest {
//    private static final Config config = new Config("config/local.config");
//    private static final String DATASETS = config.getString("input.datasets");
//    private static final String TMP = config.getString("output.tmp");
//    public static void main(String[] args) throws Exception{
//        spam_build();
//        spam_load();
//
//    }
//    static void spam_load() throws Exception{
//        System.out.println("loading classifier");
//        Classifier classifier = new LKTBFactory().deserialize(new File(TMP,"classifier.ser"));
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
//                DataSetType.CLF_DENSE, true);
//        System.out.println("test data:");
//        System.out.println(dataSet.getMetaInfo());
//
//
//        double accuracy = Accuracy.accuracy(classifier, dataSet);
//        System.out.println(accuracy);
//        ConfusionMatrix confusionMatrix = new ConfusionMatrix(classifier,dataSet);
//        System.out.println("confusion matrix:");
//        System.out.println(confusionMatrix.printWithExtLabels());
//
//
//    }
//
//    static void spam_build() throws Exception{
//
//
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
//                DataSetType.CLF_DENSE,true);
//        System.out.println(dataSet.getMetaInfo());
//
//        TrainConfig trainConfig = new LKTBTrainConfig()
//                .setLearningRate(0.1)
//                .setNumIterations(1000)
//                .setNumLeaves(7);
//        Classifier classifier = new LKTBFactory().train(dataSet,trainConfig);
//
//
//        double accuracy = Accuracy.accuracy(classifier,dataSet);
//        System.out.println("accuracy="+accuracy);
//
//
//        classifier.serialize(new File(TMP,"classifier.ser"));
//    }
//
//}