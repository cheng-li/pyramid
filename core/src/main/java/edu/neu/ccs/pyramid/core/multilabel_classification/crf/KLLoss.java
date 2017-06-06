package edu.neu.ccs.pyramid.core.multilabel_classification.crf;

import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.core.optimization.Optimizable;
import edu.neu.ccs.pyramid.core.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * KL divergence between target distribution q and predicted distribution p
 * to be minimized
 * Created by chengli on 9/26/16.
 */
public class KLLoss implements Optimizable.ByGradientValue {
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
    // size num combination* num classes
    private boolean[][] comContainsLabel;
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

    // for each combination, store the sum of probabilities over all data points
    // size = num combinations
    private double[] combProbSums;

    // size = num data * num combination
    private double[][] targetDistribution;

    // marginals of the target distribution
    // size = num data * num classes
    private double[][] targetMarginals;





    public KLLoss (CMLCRF cmlcrf, MultiLabelClfDataSet dataSet, double[][] targetDistribution, double gaussianPriorVariance) {
        this.cmlcrf = cmlcrf;
        this.supportedCombinations = cmlcrf.getSupportCombinations();
        this.numSupport = cmlcrf.getNumSupports();
        this.dataSet = dataSet;
        this.numData = dataSet.getNumDataPoints();
        this.numClasses = dataSet.getNumClasses();
        this.targetDistribution = targetDistribution;
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
        this.gradient = new DenseVector(numParameters);
        this.combProbSums = new double[numSupport];
        this.initTargetMarginals();
        this.mapParameters();
        this.initComContainsLabel();
        this.mapPairToCombination();
        this.initEmpiricalCounts();


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
        // O(NdL)
        updateClassScoreMatrix();
        updateAssignmentScoreMatrix();
        updateAssignmentProbMatrix();
        updateCombProbSums();
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


    private double calGradientForLabelPair(int parameterIndex) {
        double gradient = 0.0;
        int pos = parameterIndex - numWeightsForFeatures;
        for (int matched: labelPairToCombination.get(pos)){
            gradient += combProbSums[matched];
        }
        gradient -= this.empiricalCounts[parameterIndex];
        if (regularizeAll){
            gradient += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
        }
        return gradient;
    }


    // this calculation uses a short cut for equation (4) of
    // the paper "Collective Multi-Label Classification"
    // the sum of y can be pushed in and gives the marginal
    private double calGradientForFeature(int parameterIndex) {
        double gradient = 0.0;
        int classIndex = parameterToClass[parameterIndex];
        int featureIndex = parameterToFeature[parameterIndex];

        if (featureIndex == -1) {
            for (int i=0; i<dataSet.getNumDataPoints(); i++) {
                gradient += this.classProbMatrix[i][classIndex];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(featureIndex);
            for (Vector.Element element: featureColumn.nonZeroes()) {
                int dataPointIndex = element.index();
                double featureValue = element.get();
                gradient += this.classProbMatrix[dataPointIndex][classIndex] * featureValue;
            }
        }

        gradient -= this.empiricalCounts[parameterIndex];

        // regularize
        if (regularizeAll){
            gradient += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
        } else {
            if (featureIndex != -1) {
                gradient += cmlcrf.getWeights().getWeightForIndex(parameterIndex)/gaussianPriorVariance;
            }
        }
        return gradient;
    }



    private void initEmpiricalCounts(){
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
        }
    }



    private double calEmpiricalCountForLabelPair(int parameterIndex) {
        double empiricalCount = 0.0;
        int pos = parameterIndex - numWeightsForFeatures;
        List<Integer> comIndices = labelPairToCombination.get(pos);
        for (int i=0; i<dataSet.getNumDataPoints(); i++) {
            for (int matchedCom: comIndices){
               empiricalCount += targetDistribution[i][matchedCom];
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
                empiricalCount += targetMarginals[i][classIndex];
            }
        } else{
            Vector column = dataSet.getColumn(featureIndex);
            for (Vector.Element element: column.nonZeroes()){
                int dataIndex = element.index();
                double featureValue = element.get();
                empiricalCount += featureValue*targetMarginals[dataIndex][classIndex];
            }
        }
        return empiricalCount;
    }

    private void mapParameters() {
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

    }

    private void initComContainsLabel(){
        comContainsLabel = new boolean[numSupport][numClasses];
        for (int num=0; num< numSupport; num++) {
            for (int l=0; l<numClasses; l++) {
                if (supportedCombinations.get(num).matchClass(l)) {
                    comContainsLabel[num][l] = true;
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


        this.value = getValueForAllData() + getPenalty();
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
    }

    // NLL
    private double getValueForOneData(int i) {
        double sum = 0.0;
        // sum logZ(x_n)
        sum += MathUtil.logSumExp(combScoreMatrix[i]);
        double[] scores = combScoreMatrix[i];
        double[] targetProbs = targetDistribution[i];
        for (int j=0;j<numSupport;j++){
            sum -= scores[j]*targetProbs[j];
        }
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


    public double getPenalty(){
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

        }
        return weightSquare/(2*gaussianPriorVariance);
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


    private void mapPairToCombination(){
        this.labelPairToCombination = new ArrayList<>();
        for (int i=0;i< numWeightsForLabelPairs;i++){
            labelPairToCombination.add(new ArrayList<>());
        }
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
                case 0: if (!comContainsLabel[c][l1] && !comContainsLabel[c][l2]) list.add(c);
                    break;
                // l1 = 1; l2 = 0;
                case 1: if (comContainsLabel[c][l1] && !comContainsLabel[c][l2]) list.add(c);
                    break;
                // l1 = 0; l2 = 1;
                case 2: if (!comContainsLabel[c][l1] && comContainsLabel[c][l2]) list.add(c);
                    break;
                // l1 = 1; l2 = 1;
                case 3: if (comContainsLabel[c][l1] && comContainsLabel[c][l2]) list.add(c);
                    break;
                default: throw new RuntimeException("feature case :" + featureCase + " failed.");
            }
        }
    }

    private void updateCombProbSums(int combinationIndex){
        double sum =0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            sum += combProbMatrix[i][combinationIndex];
        }
        combProbSums[combinationIndex] = sum;
    }

    private void updateCombProbSums(){
        IntStream.range(0,numSupport).parallel()
                .forEach(this::updateCombProbSums);
    }

    private void initTargetMarginals(){
        this.targetMarginals = new double[numData][numClasses];
        IntStream.range(0, numData).parallel().forEach(this::initTargMarginals);
    }

    private void initTargMarginals(int dataPoint){
        double[] joint = targetDistribution[dataPoint];
        for (int c=0;c<joint.length;c++){
            MultiLabel multiLabel = supportedCombinations.get(c);
            double prob = joint[c];
            for (int l:multiLabel.getMatchedLabels()){
                targetMarginals[dataPoint][l] += prob;
            }
        }
    }
}
