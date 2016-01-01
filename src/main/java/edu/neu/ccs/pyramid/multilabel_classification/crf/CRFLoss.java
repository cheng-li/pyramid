package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 12/13/15.
 */
public class CRFLoss implements Optimizable.ByGradientValue {
    private static final Logger logger = LogManager.getLogger();
    private CMLCRF cmlcrf;
    private List<MultiLabel> supportedCombinations;
    private int numSupport;
    private MultiLabelClfDataSet dataSet;
    private double gaussianPriorVariance;
    private int numClasses;
    private int numParameters;
    private int numWeightsForFeatures;
    private int numWeightsForLabelPairs;
    private Vector gradient;
    private double value;
    private double[] empiricalCounts;
    private int[] parameterToL1;
    private int[] parameterToL2;
    private int[] parameterToClass;
    private int[] parameterToFeature;
    // whether the support combination contains the label;
    private boolean[][] labelInSupported;
    private boolean isParallel = true;
    private boolean isGradientCacheValid = false;
    private boolean isValueCacheValid = false;

    // numDataPoints by numClasses;
    private double[][] classScoreMatrix;

    // numDataPoints by numClasses;
    private double[][] classProbMatrix;

    // numDataPoints by numCombinations
    private double[][] combProbMatrix;

    // numDataPoints by numCombinations
    private double[][] combScoreMatrix;

    private int numData;

    //for each label pair (index), map to the list of matched combinations (index)
    // number of pairs * variable length
    private List<List<Integer>> labelPairToCombination;

    // if true, regularize all weights
    private boolean regularizeAll = false;






    public CRFLoss (CMLCRF cmlcrf, MultiLabelClfDataSet dataSet, double gaussianPriorVariance) {
        this.cmlcrf = cmlcrf;
        this.supportedCombinations = cmlcrf.getSupportCombinations();
        this.numSupport = cmlcrf.getNumSupports();
        this.dataSet = dataSet;
        this.numData = dataSet.getNumDataPoints();
        this.numClasses = dataSet.getNumClasses();
        this.gaussianPriorVariance = gaussianPriorVariance;
        this.numParameters = cmlcrf.getWeights().totalSize();
        this.numWeightsForFeatures = cmlcrf.getWeights().getNumWeightsForFeatures();
        this.numWeightsForLabelPairs = cmlcrf.getWeights().getNumWeightsForLabels();
        this.classScoreMatrix = new double[numData][numClasses];
        this.classProbMatrix = new double[numData][numClasses];
        this.combScoreMatrix = new double[numData][numSupport];
        this.combProbMatrix = new double[numData][numSupport];
        this.isGradientCacheValid = false;
        this.isValueCacheValid = false;
        this.empiricalCounts = new double[numParameters];
        this.initCache();
        this.updateEmpiricalCounts();
        this.gradient = new DenseVector(numParameters);
        this.labelPairToCombination = new ArrayList<>();
        for (int i=0;i< numWeightsForLabelPairs;i++){
            labelPairToCombination.add(new ArrayList<>());
        }
        this.mapPairToCombination();

    }

    public void setRegularizeAll(boolean regularizeAll) {
        this.regularizeAll = regularizeAll;
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
        if (logger.isDebugEnabled()){
            logger.debug("start method getGradient()");
        }
        updateClassScoreMatrix();
        updateAssignmentScoreMatrix();
        updateAssignmentProbMatrix();
        updateClassProbMatrix();
        updateGradient();
        this.isGradientCacheValid = true;
        if (logger.isDebugEnabled()){
            logger.debug("finish method getGradient()");
        }
        return this.gradient;
    }




    private void updateGradient() {
        if (logger.isDebugEnabled()){
            logger.debug("start method updateGradient()");
        }

        updatedFeatureLabelGradient();
        if (cmlcrf.considerPair()){
            updateLabelLabelGradient();
        }
        if (cmlcrf.isConsiderBmm()) {
            updateBMMGradient();
        }

        if (logger.isDebugEnabled()){
            logger.debug("finish method updateGradient()");
        }
    }


    private void updatedFeatureLabelGradient(){
        if (logger.isDebugEnabled()){
            logger.debug("start method updatedFeatureLabelGradient()");
        }
        IntStream.range(0,numWeightsForFeatures).parallel()
                .forEach(i -> gradient.set(i,calGradientForFeature(i)));
        if (logger.isDebugEnabled()){
            logger.debug("finish method updatedFeatureLabelGradient()");
        }
    }


    private void updateLabelLabelGradient(){
        if (logger.isDebugEnabled()){
            logger.debug("start method updateLabelLabelGradient()");
        }
        IntStream.range(numWeightsForFeatures,numWeightsForFeatures+numWeightsForLabelPairs).parallel()
                .forEach(i -> gradient.set(i,calGradientForLabelPair(i)));
        if (logger.isDebugEnabled()){
            logger.debug("finish method updateLabelLabelGradient()");
        }
    }

    private void updateBMMGradient(){
        int parameterIndex = numWeightsForFeatures+numWeightsForLabelPairs;
        gradient.set(parameterIndex,calGradientForBMM(parameterIndex));
    }


    private double calGradientForLabelPair(int parameterIndex) {
        double count = 0.0;
        int pos = parameterIndex - numWeightsForFeatures;
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            double[] probs = this.combProbMatrix[i];
            for (int matched: labelPairToCombination.get(pos)){
                count += probs[matched];
            }
        }
        count -= this.empiricalCounts[parameterIndex];
        if (regularizeAll){
            count += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
        }
        return count;
    }


    private double calGradientForFeature(int parameterIndex) {
        double count = 0.0;
        int classIndex = parameterToClass[parameterIndex];
        int featureIndex = parameterToFeature[parameterIndex];

        if (featureIndex == -1) {
            for (int i=0; i<dataSet.getNumDataPoints(); i++) {
                count += this.classProbMatrix[i][classIndex];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()) {
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += this.classProbMatrix[dataPointIndex][classIndex] * featureValue;
            }
        }

        count -= this.empiricalCounts[parameterIndex];

        // regularize
        if (regularizeAll){
            count += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
        } else {
            if (featureIndex != -1) {
                count += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
            }
        }
        return count;
    }


    private double calGradientForBMM(int parameterIndex){
        double g =  calExpCountForBMM() - empiricalCounts[parameterIndex];
        if (regularizeAll){
            g += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
        }
        return g;
    }

    private double calExpCountForBMM(int i){
        double sum = 0;
        double[] probs = combProbMatrix[i];
        for (int s = 0;s< numSupport;s++){
            sum += cmlcrf.mixtureScores[s]*probs[s];
        }
        return sum;
    }

    private double calExpCountForBMM(){
        return IntStream.range(0,dataSet.getNumDataPoints()).mapToDouble(this::calExpCountForBMM).sum();
    }

    private void updateEmpiricalCounts(){
        IntStream intStream;
        if (isParallel){
            intStream = IntStream.range(0, numParameters).parallel();
        } else {
            intStream = IntStream.range(0, numParameters);
        }
        intStream.forEach(this::calEmpiricalCount);
    }

    private void calEmpiricalCount(int parameterIndex) {
        if (parameterIndex < numWeightsForFeatures) {
            this.empiricalCounts[parameterIndex] = calEmpiricalCountForFeature(parameterIndex);
        } else if(parameterIndex <numWeightsForFeatures+ numWeightsForLabelPairs) {
            this.empiricalCounts[parameterIndex] = calEmpiricalCountForLabelPair(parameterIndex);
        } else {
            this.empiricalCounts[parameterIndex] = calEmpiricalCountForBMM();
        }
    }

    private double calEmpiricalCountForBMM(){
        double count =0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            int combinationIndex = cmlcrf.labelComIndices[i];
            count += cmlcrf.mixtureScores[combinationIndex];
        }
        return count;
    }

    private double calEmpiricalCountForLabelPair(int parameterIndex) {
        double empiricalCount = 0.0;
        int start = parameterIndex - numWeightsForFeatures;
        int l1 = parameterToL1[start];
        int l2 = parameterToL2[start];
        int featureCase = start % 4;
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            MultiLabel label = dataSet.getMultiLabels()[i];
            switch (featureCase) {
                // both l1, l2 equal 0;
                case 0: if (!label.matchClass(l1) && !label.matchClass(l2)) empiricalCount += 1.0;
                    break;
                // l1 = 1; l2 = 0;
                case 1: if (label.matchClass(l1) && !label.matchClass(l2)) empiricalCount += 1.0;
                    break;
                // l1 = 0; l2 = 1;
                case 2: if (!label.matchClass(l1) && label.matchClass(l2)) empiricalCount += 1.0;
                    break;
                // l1 = 1; l2 = 1;
                case 3: if (label.matchClass(l1) && label.matchClass(l2)) empiricalCount += 1.0;
                    break;
                default: throw new RuntimeException("feature case :" + featureCase + " failed.");
            }
        }
        return empiricalCount;
    }


    private double calEmpiricalCountForFeature(int parameterIndex) {
        double empiricalCount = 0.0;
        int classIndex = parameterToClass[parameterIndex];
        int featureIndex = parameterToFeature[parameterIndex];
        if (featureIndex==-1){
            for (int i=0; i<dataSet.getNumDataPoints(); i++) {
                if (dataSet.getMultiLabels()[i].matchClass(classIndex)) {
                    empiricalCount += 1;
                }
            }
        } else{
            Vector column = dataSet.getColumn(featureIndex);
            MultiLabel[] multiLabels = dataSet.getMultiLabels();
            for (Vector.Element element: column.nonZeroes()){
                int dataIndex = element.index();
                double featureValue = element.get();
                if (multiLabels[dataIndex].matchClass(classIndex)){
                    empiricalCount += featureValue;
                }
            }
        }
        return empiricalCount;
    }

    public void initCache() {
        parameterToL1 = new int[numWeightsForLabelPairs];
        parameterToL2 = new int[numWeightsForLabelPairs];
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

        labelInSupported = new boolean[numSupport][numClasses];
        for (int num=0; num< numSupport; num++) {
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

        if (regularizeAll){
            for (int k=0; k<numClasses; k++) {
                double bias = cmlcrf.getWeights().getBiasForClass(k);
                weightSquare += bias*bias;
            }

            Vector labelPairVector = cmlcrf.getWeights().getAllLabelPairWeights();
            weightSquare += labelPairVector.dot(labelPairVector);

            double bmmWeight = cmlcrf.getWeights().getAllWeights().get(numParameters-1);
            weightSquare += bmmWeight*bmmWeight;
        }

        this.value = getValueForAllData() + weightSquare/2*gaussianPriorVariance;
        this.isValueCacheValid = true;
        return this.value;
    }



    private double getValueForAllData() {
        updateClassScoreMatrix();
        updateAssignmentScoreMatrix();
        IntStream intStream;
        if (isParallel) {
            intStream = IntStream.range(0,dataSet.getNumDataPoints()).parallel();
        } else {
            intStream = IntStream.range(0,dataSet.getNumDataPoints());
        }

        return intStream.mapToDouble(this::getValueForOneData).sum();

//        return dataSetLogLikelihood(dataSet)*-1;
    }

    // NLL
    private double getValueForOneData(int i) {
        double sum = 0.0;
        // sum logZ(x_n)
        sum += MathUtil.logSumExp(combScoreMatrix[i]);
        // score for the true combination
        sum -= combScoreMatrix[i][cmlcrf.labelComIndices[i]];
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
        this.cmlcrf.updateCombLabelPartScores();
    }

    @Override
    public void setParallelism(boolean isParallel) {
        this.isParallel = isParallel;
    }

    @Override
    public boolean isParallel() {
        return this.isParallel;
    }


    private void updateClassScoreMatrix(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateClassScoreMatrix()");
        }
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i-> classScoreMatrix[i] = cmlcrf.predictClassScores(dataSet.getRow(i)));
        if (logger.isDebugEnabled()){
            logger.debug("finish updateClassScoreMatrix()");
        }
    }

    private void updateAssignmentScoreMatrix(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateAssignmentScoreMatrix()");
        }
        //todo
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> combScoreMatrix[i] = cmlcrf.predictCombinationScores(classScoreMatrix[i]));
        if (logger.isDebugEnabled()){
            logger.debug("finish updateAssignmentScoreMatrix()");
        }
    }

    private void updateAssignmentProbMatrix(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateAssignmentProbMatrix()");
        }
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> combProbMatrix[i] = cmlcrf.predictCombinationProbs(combScoreMatrix[i]));
        if (logger.isDebugEnabled()){
            logger.debug("finish updateAssignmentProbMatrix()");
        }
    }

    private void updateClassProbMatrix(){
        if (logger.isDebugEnabled()){
            logger.debug("start updateClassProbMatrix()");
        }
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> classProbMatrix[i] = cmlcrf.calClassProbs(combProbMatrix[i]));
        if (logger.isDebugEnabled()){
            logger.debug("finish updateClassProbMatrix()");
        }
    }



//    double logLikelihood(int dataPoint){
//        Vector vector = dataSet.getRow(dataPoint);
//        double[] combinationScores = cmlcrf.predictCombinationScores(vector);
//        double logDenominator = MathUtil.logSumExp(combinationScores);
//
//        double logNumerator = combinationScores[cmlcrf.labelComIndices[dataPoint]];
//        return logNumerator-logDenominator;
//    }
//
//
//    double dataSetLogLikelihood(MultiLabelClfDataSet dataSet){
//        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
//                .mapToDouble(i -> logLikelihood(i))
//                .sum();
//    }

    private void mapPairToCombination(){
        IntStream.range(0, numWeightsForLabelPairs).parallel().forEach(this::mapPairToCombination);
    }

    private void mapPairToCombination(int position) {
        List<Integer> list = labelPairToCombination.get(position);
        int l1 = parameterToL1[position];
        int l2 = parameterToL2[position];
        int featureCase = position % 4;
        for (int c=0; c< numSupport; c++) {
            switch (featureCase) {
                // both l1, l2 equal 0;
                case 0: if (!labelInSupported[c][l1] && !labelInSupported[c][l2]) list.add(c);
                    break;
                // l1 = 1; l2 = 0;
                case 1: if (labelInSupported[c][l1] && !labelInSupported[c][l2]) list.add(c);
                    break;
                // l1 = 0; l2 = 1;
                case 2: if (!labelInSupported[c][l1] && labelInSupported[c][l2]) list.add(c);
                    break;
                // l1 = 1; l2 = 1;
                case 3: if (labelInSupported[c][l1] && labelInSupported[c][l2]) list.add(c);
                    break;
                default: throw new RuntimeException("feature case :" + featureCase + " failed.");
            }
        }
    }

}
