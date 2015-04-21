package edu.neu.ccs.pyramid.experiment;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegressionInspector;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.feature.TopFeatures;
import edu.neu.ccs.pyramid.util.Serialization;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * run logistic regression on all trec8 queries
 * Created by chengli on 4/18/15.
 */
public class Exp85 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);


        if (config.getBoolean("train")){
            train(config);
        }


        if (config.getBoolean("test")){
            test(config);
        }



    }

    public static void train(Config config) throws Exception{
        int[] goodQueries = {301, 304, 306, 307, 311, 313, 318, 319, 321, 324, 331, 332, 343, 346, 347, 352, 353, 354, 357, 360, 367, 370, 374, 376, 383, 389, 390, 391, 392, 395, 398, 399, 400, 401, 404, 408, 412, 415, 418, 422, 424, 425, 426, 428, 431, 434, 435, 436, 438, 439, 443, 446, 450};
        Set<Integer> goodQuerySet = Arrays.stream(goodQueries).mapToObj(i -> i).
                collect(Collectors.toSet());



        for (int qid=401;qid<=450;qid++){
            if (goodQuerySet.contains(qid)){
                System.out.println("=============================");
                System.out.println("qid = "+qid);
                Config perQidConfig = perQidConfig(config,qid);
                System.out.println(perQidConfig);
                Exp70.train(perQidConfig);
            }
        }
    }

    public static void test(Config config) throws Exception{
        int[] goodQueries = {301, 304, 306, 307, 311, 313, 318, 319, 321, 324, 331, 332, 343, 346, 347, 352, 353, 354, 357, 360, 367, 370, 374, 376, 383, 389, 390, 391, 392, 395, 398, 399, 400, 401, 404, 408, 412, 415, 418, 422, 424, 425, 426, 428, 431, 434, 435, 436, 438, 439, 443, 446, 450};
        Set<Integer> goodQuerySet = Arrays.stream(goodQueries).mapToObj(i -> i).
                collect(Collectors.toSet());



        for (int qid=401;qid<=450;qid++){
            if (goodQuerySet.contains(qid)){
                System.out.println("=============================");
                System.out.println("qid = "+qid);
                Config perQidConfig = perQidConfig(config,qid);
                Exp70.test(perQidConfig);
            }

        }

        List<Double> accuracies = new ArrayList<>();
        List<Double> precisions = new ArrayList<>();
        List<Double> recalls = new ArrayList<>();
        List<Double> f1s = new ArrayList<>();
        List<Double> aucs = new ArrayList<>();
        List<Double> ndcgs = new ArrayList<>();
        for (int qid=401;qid<=450;qid++){
            if (goodQuerySet.contains(qid)){
                System.out.println("=============================");
                System.out.println("qid = "+qid);
                Config perQidConfig = perQidConfig(config,qid);
                File bestModelFolder = new File(perQidConfig.getString("output.folder"),"best");
                File model = new File(bestModelFolder,"model");
                LogisticRegression logisticRegression = (LogisticRegression)Serialization.deserialize(model);

                String input = perQidConfig.getString("input.folder");
                ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, perQidConfig.getString("input.testData")),
                        DataSetType.CLF_SPARSE, true);
                double accuracy = Accuracy.accuracy(logisticRegression,dataSet);
                double precision = Precision.precision(logisticRegression,dataSet,1);
                double recall = Recall.recall(logisticRegression,dataSet,1);
                double f1 = FMeasure.f1(precision,recall);
                double auc = AUC.auc(logisticRegression,dataSet);
                int[] labels = dataSet.getLabels();
                double[] scores = IntStream.range(0,dataSet.getNumDataPoints())
                        .mapToDouble(i -> logisticRegression.predictClassScore(dataSet.getRow(i),1))
                        .toArray();
                double ndcg = NDCG.ndcg(labels,scores);

                accuracies.add(accuracy);
                precisions.add(precision);
                recalls.add(recall);
                f1s.add(f1);
                aucs.add(auc);
                ndcgs.add(ndcg);
                System.out.println("accuracy = "+accuracy);
                System.out.println("precision = "+precision);
                System.out.println("recall = "+recall);
                System.out.println("auc = "+auc);
                System.out.println("ndcg = "+ndcg);


                List<TopFeatures> topFeaturesList = IntStream.range(0, logisticRegression.getNumClasses())
                        .mapToObj(k -> LogisticRegressionInspector.topFeatures(logisticRegression, k, 100000))
                        .collect(Collectors.toList());
                ObjectMapper mapper = new ObjectMapper();
                File file = new File(perQidConfig.getString("output.folder"),"topFeatures.json");
                System.out.println("writing to "+file.getAbsolutePath());
                mapper.writeValue(file, topFeaturesList);
            }

        }



        System.out.println("all done");
        System.out.println("average accuracy = "+accuracies.stream()
                .mapToDouble(acc -> acc)
                .average().getAsDouble());


        System.out.println("average precision = " + precisions.stream()
                .mapToDouble(num -> num)
                .average().getAsDouble());

        System.out.println("average recall = "+recalls.stream()
                .mapToDouble(num -> num)
                .average().getAsDouble());

        System.out.println("average f1 = "+f1s.stream()
                .mapToDouble(num -> num)
                .average().getAsDouble());

        System.out.println("average auc = "+aucs.stream()
                .mapToDouble(num -> num)
                .average().getAsDouble());

        System.out.println("average ndcg = "+ndcgs.stream()
                .mapToDouble(num -> num)
                .average().getAsDouble());
    }

    static Config perQidConfig(Config config, int qid){
        Config perQidConfig = new Config();
        Config.copy(config,perQidConfig);
        perQidConfig.setInt("qid",qid);
        String input = config.getString("input.folder");
        String perQidInput = (new File(input,""+qid)).getAbsolutePath();
        perQidConfig.setString("input.folder",perQidInput);

        String archive = config.getString("output.folder");
        String qidArchive = (new File(archive,""+qid)).getAbsolutePath();
        perQidConfig.setString("output.folder",qidArchive);

        return perQidConfig;
    }
}
