package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.util.ArgSort;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;

/**
 * The posterior is a relative notion. It is hard to set a absolute threshold. It could happen that all ProbYGivenComponent(k) are small
    We wish to find large logProportions[k] + logYGivenComponent(k)
 in the softmax function, any item smaller than the max value by more than 20 will be mapped to almost 0
 we use this to prune some components

 double[] posteriorMembership(){
 double[] logNumerator = new double[numComponents];
 for (int k=0;k<numComponents;k++){
 logNumerator[k] = logProportions[k] + logYGivenComponent(k);
 }
 return MathUtil.softmax(logNumerator);
 }
 We check components in decreasing logProportions.
 * Created by chengli on 3/25/17.
 */
public class ShortCircuitPosterior {
    int numLabels;
    int numComponents;
    // log(z=k)
    double[] logProportions;

    double[] logYGivenComponent;
    MultiLabel y;
    CBM cbm;
    Vector x;
    double skipThreshold = 30;

    public ShortCircuitPosterior(CBM cbm, Vector x, MultiLabel y) {
        this.numLabels = cbm.numLabels;
        this.y = y;
        this.x = x;
        this.cbm = cbm;
        this.numComponents = cbm.numComponents;
        this.logProportions = cbm.multiClassClassifier.predictLogClassProbs(x);
        this.logYGivenComponent = new double[numComponents];
        double max = Double.NEGATIVE_INFINITY;
        int[] sortedComponents = ArgSort.argSortDescending(logProportions);


        for (int k: sortedComponents){
            if (logProportions[k]>max-skipThreshold){
                logYGivenComponent[k] = computeLogYGivenComponent(k, max);
                double s = logProportions[k]+logYGivenComponent[k];
                if (s>max){
                    max = s;
                }
            }
        }
    }

    // the more terms we add, the smaller the sum is
    // we can stop the computation when the sum is small enough to conclude that the component is useless
    private double computeLogYGivenComponent(int k, double max){
        double sum = 0;
        // try the most likely short circuits first: positive label and prior
        for (int l: y.getMatchedLabels()){
            if (cbm.binaryClassifiers[k][l] instanceof PriorProbClassifier){
                // cheap
                sum += cbm.binaryClassifiers[k][l].predictLogClassProbs(x)[1];
                //short circuit
                if (sum + logProportions[k] < max - skipThreshold){
                    return sum;
                }
            }
        }

        // try the rest of the prior classifiers
        for (int l=0;l<numLabels;l++){
            if (cbm.binaryClassifiers[k][l] instanceof PriorProbClassifier){
                // cheap
                double[] logProbs = cbm.binaryClassifiers[k][l].predictLogClassProbs(x);

                if (y.matchClass(l)){
                    sum += logProbs[1];
                } else {
                    sum += logProbs[0];
                }
                //short circuit
                if (sum + logProportions[k] < max - skipThreshold){
                    return sum;
                }
            }
        }

        // consider all standard binary classifiers
        for (int l=0;l<numLabels;l++){
            if (!(cbm.binaryClassifiers[k][l] instanceof PriorProbClassifier)){
                // expensive
                double[] logProbs = cbm.binaryClassifiers[k][l].predictLogClassProbs(x);

                if (y.matchClass(l)){
                    sum += logProbs[1];
                } else {
                    sum += logProbs[0];
                }
                //short circuit
                if (sum + logProportions[k] < max - skipThreshold){
                    return sum;
                }
            }
        }
        return sum;
    }




    public double[] posteriorMembership(){
        double[] logNumerator = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            logNumerator[k] = logProportions[k] + logYGivenComponent[k];
        }
        return MathUtil.softmax(logNumerator);
    }
}
