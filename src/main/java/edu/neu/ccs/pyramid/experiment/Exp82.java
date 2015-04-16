package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.ClfDataSetBuilder;
import edu.neu.ccs.pyramid.eval.Accuracy;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;

/**
 * Created by chengli on 4/15/15.
 */
public class Exp82 {
    public static void main(String[] args) {
        test1();


    }

    // standard ngram
    private static void test1(){
        String pos = "I recommend it";
        String neg = "I do not recommend it";

        String feature0 = "recommend";
        String feature1 = "not recommend";

        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(2).numFeatures(2).numClasses(2)
                .build();
        dataSet.setFeatureValue(0,0,1);
        dataSet.setFeatureValue(1,0,1);
        dataSet.setFeatureValue(1,1,1);
        dataSet.setLabel(0,1);
        dataSet.setLabel(1,0);
        LogisticRegression logisticRegression = new LogisticRegression(2,2);
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression,dataSet)
                .setRegularization(1.0E-2).setL1Ratio(0).build();
        trainer.train();
        System.out.println(logisticRegression.getWeights().getBiasForClass(0));
        System.out.println(logisticRegression.getWeights().getBiasForClass(1));
        System.out.println(logisticRegression.getWeights().getWeightsWithoutBiasForClass(0));
        System.out.println(logisticRegression.getWeights().getWeightsWithoutBiasForClass(1));
        Vector vector = new DenseVector(2);
        System.out.println(Arrays.toString(logisticRegression.predictClassProbs(vector)));
        System.out.println(Accuracy.accuracy(logisticRegression,dataSet));
    }


    // maximal ngram
    private static void test2(){
        String pos = "I recommend it";
        String neg = "I do not recommend it";

        String feature0 = "recommend";
        String feature1 = "not recommend";

        ClfDataSet dataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(2).numFeatures(2).numClasses(2)
                .build();
        dataSet.setFeatureValue(0,0,1);
        dataSet.setFeatureValue(1,1,1);
        dataSet.setLabel(0,1);
        dataSet.setLabel(1,0);
        LogisticRegression logisticRegression = new LogisticRegression(2,2);
        ElasticNetLogisticTrainer trainer = ElasticNetLogisticTrainer.newBuilder(logisticRegression,dataSet)
                .setRegularization(1.0E-2).setL1Ratio(0).build();
        trainer.train();
        System.out.println(logisticRegression.getWeights().getBiasForClass(0));
        System.out.println(logisticRegression.getWeights().getBiasForClass(1));
        System.out.println(logisticRegression.getWeights().getWeightsWithoutBiasForClass(0));
        System.out.println(logisticRegression.getWeights().getWeightsWithoutBiasForClass(1));
        Vector vector = new DenseVector(2);
        System.out.println(Arrays.toString(logisticRegression.predictClassProbs(vector)));
        System.out.println(Accuracy.accuracy(logisticRegression,dataSet));
    }

}
