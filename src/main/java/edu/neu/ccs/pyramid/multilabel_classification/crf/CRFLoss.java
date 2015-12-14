package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.ProbabilityMatrix;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 12/13/15.
 */
public class CRFLoss implements Optimizable.ByGradientValue {
    private CMLCRF cmlcrf;
    private List<MultiLabel> supportedCombinations;
    private MultiLabelClfDataSet dataSet;
    private double gaussianPriorVariance;
    private int numClasses;
    private int numParameters;
    private int numWeightsForFeatures;
    private int numWeightsForLabels;
    private Vector gradient;
    private double value;

    private int[] cacheToL1;
    private int[] cacheToL2;
    private int[] cacheToClass;
    private int[] cacheToFeature;

    private boolean isParallel = false;
    private boolean isGradientCacheValid = false;
    private boolean isValueCacheValid = false;

    /**
     * numDataPoints by numSupported;
     */
    private ProbabilityMatrix probabilityMatrix;

    public CRFLoss (CMLCRF cmlcrf, MultiLabelClfDataSet dataSet, double gaussianPriorVariance) {
        this.cmlcrf = cmlcrf;
        this.supportedCombinations = cmlcrf.getSupportedCombinations();
        this.dataSet = dataSet;
        this.numClasses = dataSet.getNumClasses();
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.numParameters = cmlcrf.getWeights().totalSize();
        this.numWeightsForFeatures = cmlcrf.getWeights().getNumWeightsForFeatures();
        this.numWeightsForLabels = cmlcrf.getWeights().getNumWeightsForLabels();
        this.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),cmlcrf.getNumSupported());
        this.isGradientCacheValid = false;
        this.isValueCacheValid = false;
        this.initCache();
        this.gradient = new DenseVector(numParameters);
    }



    /**
     * gradient of log likelihood?
     * @return
     */
    @Override
    public Vector getGradient() {
        if (isGradientCacheValid) {
            return this.gradient;
        }
        updateClassProbMatrix();
        updateGradient();
        this.isGradientCacheValid = true;
        return this.gradient;
    }

    private void updateClassProbMatrix(){
        IntStream intStream;
        if (isParallel){
            intStream = IntStream.range(0,dataSet.getNumDataPoints()).parallel();
        } else {
            intStream = IntStream.range(0,dataSet.getNumDataPoints());
        }
        intStream.forEach(this::updateSupportedProbs);
    }

    private void updateSupportedProbs(int dataPointIndex){
        double[] probs = cmlcrf.predictClassProbs(dataSet.getRow(dataPointIndex));
        for (int k=0;k<probs.length;k++){
            this.probabilityMatrix.setProbability(dataPointIndex,k,probs[k]);
        }
    }

    private void updateGradient() {
        IntStream intStream;
        if (isParallel) {
            intStream = IntStream.range(0, numParameters).parallel();
        } else {
            intStream = IntStream.range(0, numParameters);
        }
        intStream.forEach(i -> this.gradient.set(i, calGradient(i)));
    }

    /**
     * TODO: better version.
     * @param parameterIndex
     * @return
     */
    private double calGradient(int parameterIndex) {
        double count = 0;
        // feature index
        if (parameterIndex < numWeightsForFeatures) {
            int classIndex = cacheToClass[parameterIndex];
            int featureIndex = cacheToFeature[parameterIndex];
            for (int i=0; i< dataSet.getNumDataPoints(); i++) {
                double fValue = 0.0;
                if (featureIndex == -1) {
                    fValue = 1.0;
                } else if (dataSet.getMultiLabels()[i].matchClass(classIndex)) {
                    fValue = dataSet.getRow(i).get(featureIndex);
                }

                double sumValue = 0.0;
                double[] probs = this.probabilityMatrix.getProbabilitiesForData(i);
                if (featureIndex == -1) {
                    for (int num=0; num<probs.length; num++) {
                        sumValue += probs[num];
                    }
                } else {
                    for (int num=0; num<probs.length; num++) {
                        MultiLabel label = supportedCombinations.get(num);
                        if (label.matchClass(classIndex)) {
                            sumValue += probs[num] * dataSet.getRow(i).get(featureIndex);
                        }
                    }
                }
                count += (sumValue - fValue);
            }
            // normalize
            if (featureIndex != -1) {
                count += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
            }
        } else { // for label pair feature
            int start = parameterIndex - numWeightsForFeatures;
            int l1 = cacheToL1[start];
            int l2 = cacheToL2[start];
            int featureCase = start % 4;
            for (int i=0; i<dataSet.getNumDataPoints(); i++) {
                MultiLabel label = dataSet.getMultiLabels()[i];
                double fValue = 0.0;
                switch (featureCase) {
                    // both l1, l2 equal 0;
                    case 0: if (!label.matchClass(l1) && !label.matchClass(l2)) fValue = 1.0;
                        break;
                    // l1 = 1; l2 = 0;
                    case 1: if (label.matchClass(l1) && !label.matchClass(l2)) fValue = 1.0;
                        break;
                    // l1 = 0; l2 = 1;
                    case 2: if (!label.matchClass(l1) && label.matchClass(l2)) fValue = 1.0;
                        break;
                    // l1 = 1; l2 = 1;
                    case 3: if (label.matchClass(l1) && label.matchClass(l2)) fValue = 1.0;
                        break;
                    default: throw new RuntimeException("feature case :" + featureCase + " failed.");
                }

                double sumValue = 0.0;
                double[] probs = this.probabilityMatrix.getProbabilitiesForData(i);
                for (int num=0; num<probs.length; num++) {
                    MultiLabel label1 = supportedCombinations.get(num);
                    switch (featureCase) {
                        // both l1, l2 equal 0;
                        case 0: if (!label1.matchClass(l1) && !label1.matchClass(l2)) sumValue+=probs[num] * 1;
                            break;
                        // l1 = 1; l2 = 0;
                        case 1: if (label1.matchClass(l1) && !label1.matchClass(l2)) sumValue+=probs[num] * 1;
                            break;
                        // l1 = 0; l2 = 1;
                        case 2: if (!label1.matchClass(l1) && label1.matchClass(l2)) sumValue+=probs[num] * 1;
                            break;
                        // l1 = 1; l2 = 1;
                        case 3: if (label1.matchClass(l1) && label1.matchClass(l2)) sumValue+=probs[num] * 1;
                            break;
                        default: throw new RuntimeException("feature case :" + featureCase + " failed.");
                    }
                }
                count += (sumValue - fValue);
            }
            count += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
        }
        return count;

    }

    public void initCache() {
        int[] mapL1 = new int[numWeightsForLabels];
        int[] mapL2 = new int[numWeightsForLabels];
        int start = 0;
        for (int l1=0; l1<numClasses; l1++) {
            for (int l2=l1+1; l2<numClasses; l2++) {
                mapL1[start] = l1;
                mapL1[start+1] = l1;
                mapL1[start+2] = l1;
                mapL1[start+3] = l1;
                mapL2[start] = l2;
                mapL2[start+1] = l2;
                mapL2[start+2] = l2;
                mapL2[start+3] = l2;
                start += 4;
            }
        }
        int[] mapClass = new int[numWeightsForFeatures];
        int[] mapFeature = new int[numWeightsForFeatures];
        for (int i=0; i<numWeightsForFeatures; i++) {
            mapClass[i] = cmlcrf.getWeights().getClassIndex(i);
            mapFeature[i] = cmlcrf.getWeights().getFeatureIndex(i);
        }

        cacheToL1 = mapL1;
        cacheToL2 = mapL2;
        cacheToClass = mapClass;
        cacheToFeature = mapFeature;
    }

    /**
     * TODO: log-likelihood?
     * @return
     */
    @Override
    public double getValue() {
        if (isValueCacheValid) {
            return this.value;
        }
        double weightSquare = 0.0;
        for (int k=0; k<numClasses; k++) {
            Vector weightVector = cmlcrf.getWeights().getWeightsWithoutBiasForClass(k);
            weightSquare += weightVector.dot(weightVector);
        }
        Vector labelPairVector = cmlcrf.getWeights().getAllLabelPairWeights();
        weightSquare += labelPairVector.dot(labelPairVector);

        double sum = 0.0;
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            MultiLabel label = dataSet.getMultiLabels()[i];
            for (int l=0; l<numClasses; l++) {
                //TODO cache the bias
                sum += cmlcrf.getWeights().getBiasForClass(l);
                if (label.matchClass(l)) {
                    sum += dataSet.getRow(i).dot(cmlcrf.getWeights().getWeightsWithoutBiasForClass(l));
                }
            }
            int start = numWeightsForFeatures;
            for (int l1=0; l1<numClasses; l1++) {
                for (int l2=l1+1; l2<numClasses; l2++) {
                    if (!label.matchClass(l1) && !label.matchClass(l2)) {
                        sum += this.cmlcrf.getWeights().getWeightForIndex(start);
                    } else if (label.matchClass(l1) && !label.matchClass(l2)) {
                        sum += this.cmlcrf.getWeights().getWeightForIndex(start + 1);
                    } else if (!label.matchClass(l1) && label.matchClass(l2)) {
                        sum += this.cmlcrf.getWeights().getWeightForIndex(start + 2);
                    } else {
                        sum += this.cmlcrf.getWeights().getWeightForIndex(start + 3);
                    }
                    start += 4;
                }
            }
        }
        this.value = -sum + weightSquare/2*gaussianPriorVariance;
        this.isValueCacheValid = true;
        return this.value;
    }

    @Override
    public Vector getParameters() {
        return cmlcrf.getWeights().getAllWeights();
    }

    @Override
    public void setParameters(Vector parameters) {
        this.cmlcrf.getWeights().setWeightVector(parameters);
        this.isValueCacheValid = false;
        this.isGradientCacheValid = false;
    }

    @Override
    public void setParallelism(boolean isParallel) {
        this.isParallel = isParallel;
    }

    @Override
    public boolean isParallel() {
        return this.isParallel;
    }

}
