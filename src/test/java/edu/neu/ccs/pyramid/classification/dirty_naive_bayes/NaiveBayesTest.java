package edu.neu.ccs.pyramid.classification.dirty_naive_bayes;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Arrays;

public class NaiveBayesTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();

    }

    private static void test1() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/train.trec"),
                DataSetType.CLF_SPARSE, true);

        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File(DATASETS, "20newsgroup/1/test.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());
        System.out.println("test");
        System.out.println(testDataset.getMetaInfo());
        System.out.println("start training");
        NaiveBayes naiveBayes = NBTrainer.train(dataSet);
        System.out.println("training done");
        System.out.println(Accuracy.accuracy(naiveBayes,dataSet));
        System.out.println(Accuracy.accuracy(naiveBayes,testDataset));
    }

    private static void test2() throws Exception{
        int numFeature  = 100;
        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder().numDataPoints(11269)
                .numFeatures(numFeature).dense(false).numClasses(20).build();
        try(BufferedReader br = new BufferedReader(new FileReader("/Users/chengli/Downloads/Rui Dong_HW4/data/news/train.data"))
        ){
            String line;
            while((line = br.readLine())!=null){
                String[] split = line.split(" ");
                int i = Integer.parseInt(split[0])-1;
                int j = Integer.parseInt(split[1])-1;
                double value = Double.parseDouble(split[2]);
                if (j<numFeature){
                    dataSet.setFeatureValue(i,j,value);
                }

            }
        }

        try(BufferedReader br2 = new BufferedReader(new FileReader("/Users/chengli/Downloads/Rui Dong_HW4/data/news/train.label"))
        ){
            String line;
            int i = 0;
            while((line = br2.readLine())!=null){
                String[] split = line.split(" ");
                int j = Integer.parseInt(split[0])-1;
                dataSet.setLabel(i,j);
                i += 1;
            }
        }

        System.out.println("start training");
        NaiveBayes naiveBayes = NBTrainer.train(dataSet);
        System.out.println("training done");
        System.out.println(Accuracy.accuracy(naiveBayes,dataSet));
        System.out.println(Arrays.toString(naiveBayes.conditionals[0]));
        System.out.println(Arrays.toString(naiveBayes.predictClassScores(dataSet.getRow(0))));
        System.out.println(Arrays.toString(naiveBayes.predictClassScores(dataSet.getRow(1))));
        System.out.println(Arrays.toString(naiveBayes.predictClassScores(dataSet.getRow(2))));
        System.out.println(dataSet.getRow(2));
        int[] prediction = naiveBayes.predict(dataSet);
//        System.out.println(Arrays.toString(prediction));



    }

}