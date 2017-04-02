package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.util.MathArrays;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/1/16.
 */
public class BMDistribution {
    int numLabels;
    int numComponents;
    // log(z=k)
    double[] logProportions;
    // log p(y_l=1|z=k)
    // size = num components * num classes * 2
    double[][][] logClassProbs;


    private List<MultiLabel> support;
    // #component * #support
    double[][] normalizedLogProbs;
    private boolean ifSupport;

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

    public double[] getLogProportions() {
        return logProportions;
    }

    public double[][][] getLogClassProbs() {
        return logClassProbs;
    }

    /**
     * skip components with small contributions
     * @param cbm
     * @param x
     * @param threshold
     */
    BMDistribution(CBM cbm, Vector x, double threshold) {
        this.numLabels = cbm.numLabels;

        double[] allLogProportions = cbm.multiClassClassifier.predictLogClassProbs(x);
        double logThreshold = Math.log(threshold);
        List<Integer> activeComponents = IntStream.range(0, allLogProportions.length).filter(k->allLogProportions[k]>=logThreshold)
                .boxed().collect(Collectors.toList());
        this.numComponents = activeComponents.size();
        //todo
//        System.out.println("active components = "+numComponents);
        this.logProportions = activeComponents.stream().mapToDouble(k->allLogProportions[k]).toArray();
        this.logClassProbs = new double[numComponents][numLabels][2];
        for (int k = 0; k< numComponents; k++){
            for (int l=0;l<numLabels;l++){
                logClassProbs[k][l] = cbm.binaryClassifiers[activeComponents.get(k)][l].predictLogClassProbs(x);
            }
        }
//        System.out.println(this.toString());
    }


    BMDistribution(CBMS cbms, Vector x) {
        this.numLabels = cbms.numLabels;
        this.numComponents = cbms.numComponents;
        this.logProportions = cbms.multiClassClassifier.predictLogClassProbs(x);
        this.logClassProbs = new double[numComponents][numLabels][2];
        for (int l=0;l<numLabels;l++){
            AugmentedLR augmentedLR = cbms.getBinaryClassifiers()[l];
            double[][] lp = augmentedLR.logAugmentedProbs(x);
            for (int k=0;k<numComponents;k++){
                logClassProbs[k][l][0] = lp[k][0];
                logClassProbs[k][l][1] = lp[k][1];
            }
        }

    }




    BMDistribution(CBM cbm, Vector x, List<MultiLabel> support) {
        this.numLabels = cbm.numLabels;
        this.numComponents = cbm.numComponents;
        this.logProportions = cbm.multiClassClassifier.predictLogClassProbs(x);
        this.support = support;
        this.ifSupport = true;
        double[][] classScore = new double[numComponents][numLabels];
        // #component * #support
        this.normalizedLogProbs = new double[numComponents][support.size()];

        for (int k=0; k<numComponents; k++) {
            for (int l=0; l<numLabels;l++) {
                classScore[k][l] = ((LogisticRegression) cbm.binaryClassifiers[k][l]).predictClassScores(x)[1];
            }
            double[] supportScores = new double[support.size()];
            for (int s=0; s<support.size(); s++) {
                MultiLabel label = support.get(s);
                for (Integer l : label.getMatchedLabels()) {
                    supportScores[s] += classScore[k][l];
                }
            }
            double[] supportProbs = MathUtil.softmax(supportScores);
            for (int s=0; s<support.size(); s++) {
                normalizedLogProbs[k][s] = Math.log(supportProbs[s]);
            }
        }
    }


    // log p(y|z=k)
    private double logYGivenComponent(MultiLabel y, int k){

        if (ifSupport) {
            return logYGivenComponentBySupport(y, k);
        } else {
            return logYGivenComponentByDefault(y, k);
        }
    }

    private double logYGivenComponentBySupport(MultiLabel y, int k) {
        int supportId = -1;
        for (int l=0; l<support.size(); l++) {
            if (support.get(l).equals(y)) {
                supportId = l;
                break;
            }
        }
        return normalizedLogProbs[k][supportId];
    }

    public double logYGivenComponentByDefault(MultiLabel y, int k) {
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


    // log p(y|z=k)
    private double logYGivenComponent(MultiLabel y, int k, double[] noiseLabelWeight){
        double sum = 0.0;
        for (int l=0; l< numLabels; l++) {
            if (y.matchClass(l)) {
                sum += noiseLabelWeight[l] * logClassProbs[k][l][1];
            } else {
                sum += noiseLabelWeight[l] * logClassProbs[k][l][0];
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

    // p(z=k|y)
    double[] posteriorMembership(MultiLabel y, double[] noiseLabelWeight){
        double[] logNumerator = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            logNumerator[k] = logProportions[k] + logYGivenComponent(y, k, noiseLabelWeight);
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

    public double logProbability(MultiLabel y){
        double[] logPs = new double[numComponents];
        for (int k=0;k<numComponents;k++){
            logPs[k] = logProportions[k] + logYGivenComponent(y, k);
        }
        double logP = MathUtil.logSumExp(logPs);
//        System.out.println("log Ps = "+Arrays.toString(logPs));
//        System.out.println("log probability = "+logP);
        return logP;
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

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("BMDistribution{");
        sb.append("numLabels=").append(numLabels);
        sb.append(", numComponents=").append(numComponents);
        sb.append(", logProportions=").append(Arrays.toString(logProportions));
        sb.append(", logClassProbs=").append(Arrays.deepToString(logClassProbs));
        sb.append('}');
        return sb.toString();
    }
}
