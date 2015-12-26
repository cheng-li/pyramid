package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by chengli on 12/25/15.
 */
public class BMMInspector {
    public static double distanceFromMean(BMMClassifier bmm){
        int numClasses = bmm.getNumClasses();
        return IntStream.range(0,numClasses).mapToDouble(l -> distanceFromMean(bmm,l)).average().getAsDouble();
    }

    public static double distanceFromMean(BMMClassifier bmm, int label){
        Classifier.ProbabilityEstimator[][] logistics = bmm.getBinaryClassifiers();
        int numClusters = bmm.getNumClusters();
        int numFeatures =  ((LogisticRegression)logistics[0][0]).getNumFeatures();

        Vector positiveAverageVector = new DenseVector(numFeatures);
        for (int k=0;k<numClusters;k++){
            Vector positiveVector = ((LogisticRegression) logistics[k][label]).getWeights().getWeightsWithoutBiasForClass(1);
            positiveAverageVector = positiveAverageVector.plus(positiveVector);
        }


        positiveAverageVector = positiveAverageVector.divide(numClusters);

        double dis = 0;
        for (int k=0;k<numClusters;k++){

            Vector positiveVector = ((LogisticRegression) logistics[k][label]).getWeights().getWeightsWithoutBiasForClass(1);
            dis += positiveVector.minus(positiveAverageVector).norm(2);
        }
        return dis/numClusters;
    }
}
