package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 3/25/17.
 */
public class ShortCircuitPosterior {
    int numLabels;
    int numComponents;
    // log(z=k)
    double[] logProportions;
    // log p(y_l=1|z=k)
    // size = num components * num classes * 2
    double[][][] logClassProbs;
    MultiLabel y;
    CBM cbm;
    Vector x;

    ShortCircuitPosterior(CBM cbm, Vector x, MultiLabel y) {
        this.numLabels = cbm.numLabels;
        this.y = y;
        this.x = x;
        this.cbm = cbm;
        this.numComponents = cbm.numComponents;
        this.logProportions = cbm.multiClassClassifier.predictLogClassProbs(x);
        this.logClassProbs = new double[numComponents][numLabels][2];
        for (int k = 0; k< numComponents; k++){
            fillBinaryProbs(k);
        }
    }

    private void fillBinaryProbs(int k){
        // try the most likely short circuits first: positive label and prior
        for (int l: y.getMatchedLabels()){
            if (cbm.binaryClassifiers[k][l] instanceof PriorProbClassifier){
                // cheap
                logClassProbs[k][l] = cbm.binaryClassifiers[k][l].predictLogClassProbs(x);
                double logProb = logClassProbs[k][l][1];
                //short circuit
                if (logProb==Double.NEGATIVE_INFINITY){
                    return;
                }
            }
        }

        // try the rest of the prior classifiers
        for (int l=0;l<numLabels;l++){
            if (cbm.binaryClassifiers[k][l] instanceof PriorProbClassifier){
                // cheap
                logClassProbs[k][l] = cbm.binaryClassifiers[k][l].predictLogClassProbs(x);
                double logProb;
                if (y.matchClass(l)){
                    logProb = logClassProbs[k][l][1];
                } else {
                    logProb = logClassProbs[k][l][0];
                }
                //short circuit
                if (logProb==Double.NEGATIVE_INFINITY){
                    return;
                }
            }
        }

        // consider all standard binary classifiers
        for (int l=0;l<numLabels;l++){
            if (!(cbm.binaryClassifiers[k][l] instanceof PriorProbClassifier)){
                // expensive
                logClassProbs[k][l] = cbm.binaryClassifiers[k][l].predictLogClassProbs(x);
                double logProb;
                if (y.matchClass(l)){
                    logProb = logClassProbs[k][l][1];
                } else {
                    logProb = logClassProbs[k][l][0];
                }

                if (logProb==Double.NEGATIVE_INFINITY){
                    return;
                }
            }
        }
    }

    private double logYGivenComponent(int k) {
        double sum = 0.0;
        for (int l=0; l< numLabels; l++) {
            if (y.matchClass(l)) {
                sum += logClassProbs[k][l][1];
            } else {
                sum += logClassProbs[k][l][0];
            }
            if (sum==Double.NEGATIVE_INFINITY){
                break;
            }
        }
        return sum;
    }


    double[] posteriorMembership(){
        double[] logNumerator = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            logNumerator[k] = logProportions[k] + logYGivenComponent(k);
        }
        return MathUtil.softmax(logNumerator);
    }
}
