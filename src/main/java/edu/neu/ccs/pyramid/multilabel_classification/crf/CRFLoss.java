package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.ProbabilityMatrix;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 12/13/15.
 */
public class CRFLoss implements Optimizable.ByGradientValue {
    private CMLCRF cmlcrf;
    private List<MultiLabel> supportedCombinations;
    private int numSupported;
    private MultiLabelClfDataSet dataSet;
    private double gaussianPriorVariance;
    private int numClasses;
    private int numParameters;
    private int numWeightsForFeatures;
    private int numWeightsForLabels;
    private Vector gradient;
    private double value;

    private double[] empiricalCounts;

    private int[] parameterToL1;
    private int[] parameterToL2;
    private int[] parameterToClass;
    private int[] parameterToFeature;
    // if the supported conbination contains the label;
    private boolean[][] labelInSupported;

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
        this.numSupported = cmlcrf.getNumSupported();
        this.dataSet = dataSet;
        this.numClasses = dataSet.getNumClasses();
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.numParameters = cmlcrf.getWeights().totalSize();
        this.numWeightsForFeatures = cmlcrf.getWeights().getNumWeightsForFeatures();
        this.numWeightsForLabels = cmlcrf.getWeights().getNumWeightsForLabels();
        this.probabilityMatrix = new ProbabilityMatrix(dataSet.getNumDataPoints(),numSupported);
        this.isGradientCacheValid = false;
        this.isValueCacheValid = false;
        this.empiricalCounts = new double[numParameters];
        this.initCache();
        this.updateEmpricalCounts();
        this.gradient = new DenseVector(numParameters);
    }



    /**
     * gradient of log likelihood
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
        double[] probs = cmlcrf.predictCombinationProbs(dataSet.getRow(dataPointIndex));
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
     * return the gradient.
     * @param parameterIndex
     * @return the gradient.
     */
    private double calGradient(int parameterIndex) {
        // get gradient for feature label pair.
        if (parameterIndex < numWeightsForFeatures) {
            return calGradientForFeature(parameterIndex);
        } else if (parameterIndex <numWeightsForFeatures + numWeightsForLabels){
            // get gradient for label pair;
            return calGradientForLabelPair(parameterIndex);
        } else {
            return calGradientForBMM(parameterIndex);
        }

    }

    /**
     * return the gradient by given parameterIndex, which is label and label pair weight.
     * @param parameterIndex
     * @return gradient for label pair weight.
     */
    private double calGradientForLabelPair(int parameterIndex) {
        double count = 0.0;
        int start = parameterIndex - numWeightsForFeatures;
        int l1 = parameterToL1[start];
        int l2 = parameterToL2[start];
        int featureCase = start % 4;
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            double[] probs = this.probabilityMatrix.getProbabilitiesForData(i);
            for (int num=0; num<probs.length; num++) {
                switch (featureCase) {
                    // both l1, l2 equal 0;
                    case 0: if (!labelInSupported[num][l1] && !labelInSupported[num][l2]) count+=probs[num];
                        break;
                    // l1 = 1; l2 = 0;
                    case 1: if (labelInSupported[num][l1] && !labelInSupported[num][l2]) count+=probs[num];
                        break;
                    // l1 = 0; l2 = 1;
                    case 2: if (!labelInSupported[num][l1] && labelInSupported[num][l2]) count+=probs[num];
                        break;
                    // l1 = 1; l2 = 1;
                    case 3: if (labelInSupported[num][l1] && labelInSupported[num][l2]) count+=probs[num];
                        break;
                    default: throw new RuntimeException("feature case :" + featureCase + " failed.");
                }
            }
        }
        count -= this.empiricalCounts[parameterIndex];
        count += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
        return count;
    }

    /**
     * return the gradient by given parameterIndex, which is feature and label pair weight.
     * @param parameterIndex
     * @return gradient for feature and label pair weight.
     */
    private double calGradientForFeature(int parameterIndex) {
        double count = 0.0;
        int classIndex = parameterToClass[parameterIndex];
        int featureIndex = parameterToFeature[parameterIndex];

        if (featureIndex == -1) {
            for (int i=0; i<dataSet.getNumDataPoints(); i++) {
                double[] probs = this.probabilityMatrix.getProbabilitiesForData(i);
                for (int num=0; num<probs.length; num++) {
                    if (labelInSupported[num][classIndex]) {
                        count += probs[num];
                    }
                }
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()) {
                int dataPointIndex = element.index();
                double featureValue = element.get();
                double[] probs = this.probabilityMatrix.getProbabilitiesForData(dataPointIndex);
                for (int num=0; num<probs.length; num++) {
                    if (labelInSupported[num][classIndex]) {
                        count += probs[num] * featureValue;
                    }
                }
            }
        }

        count -= this.empiricalCounts[parameterIndex];
        // normalize
        if (featureIndex != -1) {
            count += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
        }
        return count;
    }


    private double calGradientForBMM(int parameterIndex){
        return calExpCountForBMM() - empiricalCounts[parameterIndex]
                + cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
    }

    private double calExpCountForBMM(int i){
        double sum = 0;
        double[] probs = probabilityMatrix.getProbabilitiesForData(i);
        for (int s = 0;s<numSupported;s++){
            MultiLabel multiLabel = supportedCombinations.get(s);
            sum += cmlcrf.bmm.logProbability(multiLabel.toVector(numClasses))*probs[s];
        }
        return sum;
    }

    private double calExpCountForBMM(){
        return IntStream.range(0,dataSet.getNumDataPoints()).mapToDouble(this::calExpCountForBMM).sum();
    }

    private void updateEmpricalCounts(){
        IntStream intStream;
        if (isParallel){
            intStream = IntStream.range(0, numParameters).parallel();
        } else {
            intStream = IntStream.range(0, numParameters);
        }
        intStream.forEach(i -> calEmpricalCount(i));
    }

    private void calEmpricalCount(int parameterIndex) {
        if (parameterIndex < numWeightsForFeatures) {
            this.empiricalCounts[parameterIndex] = calEmpricalCountForFeature(parameterIndex);
        } else if(parameterIndex <numWeightsForFeatures+numWeightsForLabels) {
            this.empiricalCounts[parameterIndex] = calEmpricalCountForLabelPair(parameterIndex);
        } else {
            this.empiricalCounts[parameterIndex] = calEmpiricalCountForBMM();
        }
    }

    private double calEmpiricalCountForBMM(){
        double count =0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            count += cmlcrf.bmm.logProbability(dataSet.getMultiLabels()[i].toVector(numClasses));
        }
        return count;
    }

    private double calEmpricalCountForLabelPair(int parameterIndex) {
        double empricalCount = 0.0;
        int start = parameterIndex - numWeightsForFeatures;
        int l1 = parameterToL1[start];
        int l2 = parameterToL2[start];
        int featureCase = start % 4;
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            MultiLabel label = dataSet.getMultiLabels()[i];
            switch (featureCase) {
                // both l1, l2 equal 0;
                case 0: if (!label.matchClass(l1) && !label.matchClass(l2)) empricalCount += 1.0;
                    break;
                // l1 = 1; l2 = 0;
                case 1: if (label.matchClass(l1) && !label.matchClass(l2)) empricalCount += 1.0;
                    break;
                // l1 = 0; l2 = 1;
                case 2: if (!label.matchClass(l1) && label.matchClass(l2)) empricalCount += 1.0;
                    break;
                // l1 = 1; l2 = 1;
                case 3: if (label.matchClass(l1) && label.matchClass(l2)) empricalCount += 1.0;
                    break;
                default: throw new RuntimeException("feature case :" + featureCase + " failed.");
            }
        }
        return empricalCount;
    }


    private double calEmpricalCountForFeature(int parameterIndex) {
        double empricalCount = 0.0;
        int classIndex = parameterToClass[parameterIndex];
        int featureIndex = parameterToFeature[parameterIndex];
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            double featureValue = (featureIndex==-1) ? 1.0 : dataSet.getRow(i).get(featureIndex);
            if (dataSet.getMultiLabels()[i].matchClass(classIndex)) {
                empricalCount += featureValue;
            }
        }
        return empricalCount;
    }

    public void initCache() {
        parameterToL1 = new int[numWeightsForLabels];
        parameterToL2 = new int[numWeightsForLabels];
        int start = 0;
        for (int l1=0; l1<numClasses; l1++) {
            for (int l2=l1+1; l2<numClasses; l2++) {
                parameterToL1[start] = l1;
                parameterToL1[start+1] = l1;
                parameterToL1[start+2] = l1;
                parameterToL1[start+3] = l1;
                parameterToL2[start] = l2;
                parameterToL2[start+1] = l2;
                parameterToL2[start+2] = l2;
                parameterToL2[start+3] = l2;
                start += 4;
            }
        }
        parameterToClass = new int[numWeightsForFeatures];
        parameterToFeature = new int[numWeightsForFeatures];
        for (int i=0; i<numWeightsForFeatures; i++) {
            parameterToClass[i] = cmlcrf.getWeights().getClassIndex(i);
            parameterToFeature[i] = cmlcrf.getWeights().getFeatureIndex(i);
        }

        labelInSupported = new boolean[numSupported][numClasses];
        for (int num=0; num<numSupported; num++) {
            for (int l=0; l<numClasses; l++) {
                if (supportedCombinations.get(num).matchClass(l)) {
                    labelInSupported[num][l] = true;
                }
            }
        }



    }

    /**
     * @return negative log-likelihood
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

        this.value = getValueForAllData() + weightSquare/2*gaussianPriorVariance;
        this.isValueCacheValid = true;
        return this.value;
    }



    private double getValueForAllData() {
        IntStream intStream;
        if (isParallel) {
            intStream = IntStream.range(0,dataSet.getNumDataPoints()).parallel();
        } else {
            intStream = IntStream.range(0,dataSet.getNumDataPoints());
        }

        return intStream.mapToDouble(i -> this.getValueForOneData(i)).sum();
    }

    private double getValueForOneData(int i) {
        double sum = 0.0;
        MultiLabel label = dataSet.getMultiLabels()[i];
        Vector vector = dataSet.getRow(i);
        // sum logZ(x_n)
        sum += MathUtil.logSumExp(cmlcrf.predictCombinationScores(vector));
        sum -= cmlcrf.predictCombinationScore(vector, label);
        return sum;
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
