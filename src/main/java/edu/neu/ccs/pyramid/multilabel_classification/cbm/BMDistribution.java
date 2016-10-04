package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/1/16.
 */
public class BMDistribution {
    private int numLabels;
    private int numComponents;
    // log(z=k)
    private double[] logProportions;
    // log p(y_l=1|z=k)
    // size = num components * num classes * 2
    private double[][][] logClassProbs;


    BMDistribution(CBM cbm, Vector x) {
        this.numLabels = cbm.numLabels;
        this.numComponents = cbm.numComponents;
        this.logProportions = cbm.multiClassClassifier.predictLogClassProbs(x);
        this.logClassProbs = new double[numComponents][numLabels][2];
        for (int k = 0; k< numComponents; k++){
            for (int l=0;l<numLabels;l++){
                logClassProbs[k][l] = cbm.binaryClassifiers[k][l].predictLogClassProbs(x);
            }
        }
    }

    // log p(y|z=k)
    private double logYGivenComponent(MultiLabel y, int k){
        double sum = 0.0;
        for (int l=0; l< numLabels; l++) {
            if (y.matchClass(l)) {
                sum += logClassProbs[k][l][1];
            } else {
                sum += logClassProbs[k][l][0];
            }
        }
        return sum;
    }

    // p(z=k|y)
    double[] posteriorMembership(MultiLabel y){
        double[] logNumerator = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            logNumerator[k] = logProportions[k] + logYGivenComponent(y, k);
        }
        double logDenominator = MathUtil.logSumExp(logNumerator);
        double[] membership = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            membership[k] = Math.exp(logNumerator[k]-logDenominator);
        }
        return membership;
    }

    // log p(z=k|y)
    double[] logPosteriorMembership(MultiLabel y){
        double[] logNumerator = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            logNumerator[k] = logProportions[k] + logYGivenComponent(y, k);
        }
        double logDenominator = MathUtil.logSumExp(logNumerator);
        double[] membership = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            membership[k] = logNumerator[k]-logDenominator;
        }
        return membership;
    }

    double logProbability(MultiLabel y){
        double[] logPs = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            logPs[k] = logProportions[k] + logYGivenComponent(y, k);
        }
        return MathUtil.logSumExp(logPs);
    }

    double probability(MultiLabel y){
        return Math.exp(logProbability(y));
    }

    private double marginal(int labelIndex){
        double sum = 0;
        for (int k=0;k<numComponents;k++){
            sum += Math.exp(logProportions[k])*Math.exp(logClassProbs[k][labelIndex][1]);
        }
        return sum;
    }

    double[] marginals(){
        double[] m = new double[numLabels];
        for (int l=0;l<numLabels;l++){
            m[l] = marginal(l);
        }
        return m;
    }

    List<MultiLabel> sample(int numSamples){
        List<MultiLabel> list = new ArrayList<>();
        double[] proportions = Arrays.stream(logProportions).map(Math::exp).toArray();
        double[][] classProbs = new double[numComponents][numLabels];
        for (int k = 0; k< numComponents; k++){
            for (int l=0;l<numLabels;l++){
                classProbs[k][l] = Math.exp(logClassProbs[k][l][1]);
            }
        }
        int[] components = IntStream.range(0, numComponents).toArray();
        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(components, proportions);
        BernoulliDistribution[][] bernoulliDistributions = new BernoulliDistribution[numComponents][numLabels];
        for (int k=0;k<numComponents;k++){
            for (int l=0;l<numLabels;l++){
                bernoulliDistributions[k][l] = new BernoulliDistribution(classProbs[k][l]);
            }
        }
        for (int num=0;num<numSamples;num++){
            MultiLabel multiLabel = new MultiLabel();
            int k = enumeratedIntegerDistribution.sample();
            for (int l=0; l<numLabels; l++) {
                int v = bernoulliDistributions[k][l].sample();
                if (v==1){
                    multiLabel.addLabel(l);
                }
            }
            list.add(multiLabel);
        }
        return list;
    }
}
