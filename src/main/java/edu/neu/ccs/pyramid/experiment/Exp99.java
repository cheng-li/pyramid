package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * elastic for 5folds
 * Created by chengli on 5/5/15.
 */
public class Exp99 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file..");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

//        (new File(config.getString("output.folder"))).mkdirs();

        List<Double> ridgeAccs = new ArrayList<>();
        List<Double> elasticAccs = new ArrayList<>();
        List<Double> lassoAccs = new ArrayList<>();
        for (int fold = 1;fold<=5;fold++){
//            train(config,fold);
            test(config,ridgeAccs,elasticAccs,lassoAccs,fold);
        }

        System.out.println("==========================");
        System.out.println("ridge accuracy");
        System.out.println(ridgeAccs);
        System.out.println("elastic accuracy");
        System.out.println(elasticAccs);
        System.out.println("lasso accuracy");
        System.out.println(lassoAccs);
        System.out.println("ridge mean = "+ridgeAccs.stream().mapToDouble(a -> a).average().getAsDouble());
        System.out.println("elastic mean = "+elasticAccs.stream().mapToDouble(a -> a).average().getAsDouble());
        System.out.println("lasso mean = "+lassoAccs.stream().mapToDouble(a -> a).average().getAsDouble());
    }


//    public static void train(Config config, int fold) throws Exception{
//        System.out.println("training fold "+fold);
//        String input = config.getString("input.folder");
//        File foldFolder = new File(input,"fold_"+fold);
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(foldFolder, "train.trec"),
//                DataSetType.CLF_SPARSE, true);
//
//        File outputFolder = new File(config.getString("output.folder"),"fold_"+fold);
//        outputFolder.mkdirs();
//
//        LogisticRegression ridge = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
//        ElasticNetLogisticTrainer ridgeTrainer = ElasticNetLogisticTrainer.newBuilder(ridge,dataSet)
//                .setEpsilon(0.01).setL1Ratio(0).setRegularization(config.getDouble("ridge.reg")).build();
//        ridgeTrainer.train();
//
//        System.out.println("ridge training acc ="+Accuracy.accuracy(ridge,dataSet));
//        Serialization.serialize(ridge,new File(outputFolder,"ridge.ser"));
//
//
//        LogisticRegression elastic = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
//        ElasticNetLogisticTrainer elasticTrainer = ElasticNetLogisticTrainer.newBuilder(elastic,dataSet)
//                .setEpsilon(0.01).setL1Ratio(0.1).setRegularization(config.getDouble("elastic.reg")).build();
//        elasticTrainer.train();
//        System.out.println("elastic training acc ="+Accuracy.accuracy(elastic,dataSet));
//        Serialization.serialize(elastic,new File(outputFolder,"elastic.ser"));
//
//
//        LogisticRegression lasso = new LogisticRegression(dataSet.getNumClasses(),dataSet.getNumFeatures());
//        ElasticNetLogisticTrainer lassoTrainer = ElasticNetLogisticTrainer.newBuilder(lasso,dataSet)
//                .setEpsilon(0.01).setL1Ratio(1).setRegularization(config.getDouble("lasso.reg")).build();
//        lassoTrainer.train();
//        System.out.println("lasso training acc ="+Accuracy.accuracy(lasso,dataSet));
//        Serialization.serialize(lasso,new File(outputFolder,"lasso.ser"));
//
//    }

    public static void test(Config config, List<Double> ridgeAccs, List<Double> elasticAccs,
                            List<Double> lassoAccs, int fold) throws Exception{
        String input = config.getString("input.folder");
        File foldFolder = new File(input,"fold_"+fold);
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(foldFolder, "test.trec"),
                DataSetType.CLF_SPARSE, true);
        File outputFolder = new File(config.getString("output.folder"),"fold_"+fold);

        LogisticRegression ridge = (LogisticRegression)Serialization.deserialize(new File(outputFolder,"ridge.ser"));
        LogisticRegression elastic = (LogisticRegression)Serialization.deserialize(new File(outputFolder,"elastic.ser"));
        LogisticRegression lasso = (LogisticRegression)Serialization.deserialize(new File(outputFolder,"lasso.ser"));

        ridgeAccs.add(Accuracy.accuracy(ridge,dataSet));
        elasticAccs.add(Accuracy.accuracy(elastic,dataSet));
        lassoAccs.add(Accuracy.accuracy(lasso,dataSet));
    }

}
