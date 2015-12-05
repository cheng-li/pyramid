//package edu.neu.ccs.pyramid.classification.ecoc;
//
//import edu.neu.ccs.pyramid.classification.ClassifierFactory;
//import edu.neu.ccs.pyramid.classification.TrainConfig;
//import edu.neu.ccs.pyramid.classification.lkboost.LKTBFactory;
//import edu.neu.ccs.pyramid.classification.lkboost.LKTBTrainConfig;
//import edu.neu.ccs.pyramid.classification.lkboost.LKTreeBoost;
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.ClfDataSet;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//import edu.neu.ccs.pyramid.eval.Accuracy;
//
//import java.io.File;
//
//public class ECOCTest {
//    private static final Config config = new Config("config/local.config");
//    private static final String DATASETS = config.getString("input.datasets");
//    private static final String TMP = config.getString("output.tmp");
//    public static void main(String[] args) throws Exception{
//        test1();
//    }
//
//    public static void test1()throws Exception{
//        spam_build();
//        spam_load();
//    }
//
//    static void spam_load() throws Exception{
//        System.out.println("loading ensemble");
//        LKTreeBoost lkTreeBoost = LKTreeBoost.deserialize(new File(TMP,"/LKTreeBoostTest/ensemble.ser"));
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
//                DataSetType.CLF_DENSE, true);
//        System.out.println("test data:");
//        System.out.println(dataSet.getMetaInfo());
//
//        ECOC ecoc = ECOC.deserialize(new File(TMP,"ecoc/ecoc.ser"));
//
//        double accuracy = Accuracy.accuracy(ecoc, dataSet);
//        System.out.println(accuracy);
//    }
//
//    static void spam_build() throws Exception{
//
//
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS,"/spam/trec_data/train.trec"),
//                DataSetType.CLF_DENSE,true);
//        System.out.println(dataSet.getMetaInfo());
//
//        ClassifierFactory classifierFactory = new LKTBFactory();
//        TrainConfig trainConfig = new LKTBTrainConfig();
//        ECOCConfig ecocConfig = new ECOCConfig().setCodeType(CodeMatrix.CodeType.EXHAUSTIVE);
//        ECOC ecoc = new ECOC(ecocConfig,
//                dataSet,
//                new File(TMP,"ecoc/models").getAbsolutePath(),
//                classifierFactory,
//                trainConfig);
//        System.out.println(ecoc.getCodeMatrix().toString());
//
//        ecoc.train();
//        ecoc.serialize(new File(TMP,"ecoc/ecoc.ser"));
//    }
//
//}