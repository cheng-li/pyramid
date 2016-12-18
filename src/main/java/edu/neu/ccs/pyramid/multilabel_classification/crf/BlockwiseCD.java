package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Blockwise Coordiante Descent for CRF
 * see:
 * "Efficient Learning of Sparse Conditional Random Fields for Supervised
 * Sequence Labelling
 * by Nataliya Sokolovska et al."
 *
 * Created by Rainicy on 11/27/16.
 */
public class BlockwiseCD {

    private CMLCRF cmlcrf;
    private List<MultiLabel> supportedCombinations;
    private int numSupport;
    private MultiLabelClfDataSet dataSet;
    private double regularization;
    private double l1Ratio;
    private int numClasses;
    private int numParameters;
    private int numWeightsForFeatures;
    private int numWeightsForLabelPairs;
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


    // for each combination, store the sum of probabilities over all data points
    // size = num combinations
    private double[] combProbSums;

    // for each data point, store the position of the true combination in the support list
    private int[] labelComIndices;

    // terminator when converge
    private Terminator terminator;
    // features for dataset
    private int numFeatures;
    // labels to parameterIndex
    private int[][] labelPairToParams;
    private double weight;

    public BlockwiseCD (CMLCRF cmlcrf, MultiLabelClfDataSet dataSet) {
        this(cmlcrf, dataSet, 0.0, 1.0);
    }
    public BlockwiseCD (CMLCRF cmlcrf, MultiLabelClfDataSet dataSet, double l1Ratio, double regularization) {
        this.cmlcrf = cmlcrf;
        this.supportedCombinations = cmlcrf.getSupportCombinations();
        this.numSupport = cmlcrf.getNumSupports();
        this.dataSet = dataSet;
        this.numData = dataSet.getNumDataPoints();
        this.numClasses = dataSet.getNumClasses();
        this.l1Ratio = l1Ratio;
        this.regularization = regularization;
        this.numParameters = cmlcrf.getWeights().totalSize();
        this.numWeightsForFeatures = cmlcrf.getWeights().getNumWeightsForFeatures();
        this.numWeightsForLabelPairs = cmlcrf.getWeights().getNumWeightsForLabels();
        this.classScoreMatrix = new double[numData][numClasses];
        this.classProbMatrix = new double[numData][numClasses];
        this.combScoreMatrix = new double[numData][numSupport];
        this.combProbMatrix = new double[numData][numSupport];
        this.empiricalCounts = new double[numParameters];
        this.initCache();
        this.updateEmpiricalCounts();
        this.labelPairToCombination = new ArrayList<>();
        for (int i=0;i< numWeightsForLabelPairs;i++){
            labelPairToCombination.add(new ArrayList<>());
        }
        this.mapPairToCombination();
        this.combProbSums = new double[numSupport];

        Map<MultiLabel,Integer> map = new HashMap<>();
        for (int s=0;s< numSupport;s++){
            map.put(supportedCombinations.get(s),s);
        }
        this.labelComIndices = new int[dataSet.getNumDataPoints()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            labelComIndices[i] = map.get(dataSet.getMultiLabels()[i]);
        }

        this.terminator = new Terminator();
        this.numFeatures = dataSet.getNumFeatures();
        this.weight = 1.0/numData;
    }


    public void optimize() {
        int iter = 1;
        while (true) {
            iterate();
            double loss = getValue();
            terminator.add(loss);
            System.out.println(iter + ": " + loss);
            if(terminator.shouldTerminate()) {
                break;
            }
            iter++;
        }
    }

    public void iterate() {
//        updateClassScoreMatrix();
//        updateAssignmentScoreMatrix();
//        updateAssignmentProbMatrix();
//        updateCombProbSums();
//        updateClassProbMatrix();
        //        for (int x=0; x<(numFeatures+1); x++) {
//            updateClassScoreMatrix();
//            updateAssignmentScoreMatrix();
//            updateAssignmentProbMatrix();
//            updateCombProbSums();
//            updateClassProbMatrix();
//            for (int l=0; l<numClasses; l++) {
//                iterateForF(l*(numFeatures+1) + x);
//            }
//        }
        for (int i=0; i<numWeightsForFeatures; i++) {
            updateClassScoreMatrix();
            updateAssignmentScoreMatrix();
            updateAssignmentProbMatrix();
            updateCombProbSums();
            updateClassProbMatrix();
            iterateForF(i);
        }


        for (int i=numWeightsForFeatures; i<numParameters; i++) {
            updateClassScoreMatrix();
            updateAssignmentScoreMatrix();
            updateAssignmentProbMatrix();
            updateCombProbSums();
            updateClassProbMatrix();
            iterateForLF(i);
        }
    }

    /**
     * iterate for label-label features.
     * @param parameterIndex
     */
    private void iterateForLF(int parameterIndex) {
        int pos = parameterIndex - numWeightsForFeatures;
        double gradientForLabelPair = calGradientForLabelPair(pos);
        double hessiansForLabelPair = calHessiansForLabelPair(pos);
        double fit = this.weight * (hessiansForLabelPair * cmlcrf.getWeights().getWeightForIndex(parameterIndex) - gradientForLabelPair);
        double numerator = softThreshold(fit);
        double denominator = this.weight * hessiansForLabelPair + regularization * (1-l1Ratio);
        double newCoeff = 0;
        if (denominator != 0) {
            newCoeff = numerator / denominator;
        }
        cmlcrf.getWeights().getAllWeights().set(parameterIndex, newCoeff);

    }

    /**
     * iterate for feature-label features.
     * @param parameterIndex
     */
    private void iterateForF(int parameterIndex) {
        int classIndex = parameterToClass[parameterIndex];
        int featureIndex = parameterToFeature[parameterIndex];
        double gradientForFeature = calGradientForFeature(classIndex, featureIndex, parameterIndex);
        double hessianForFeature =  calHessiansForFeature(classIndex, featureIndex);
        double fit = this.weight * (hessianForFeature * cmlcrf.getWeights().getWeightForIndex(parameterIndex) - gradientForFeature);
        double numerator = softThreshold(fit);
        double denominator = this.weight * hessianForFeature + regularization * (1-l1Ratio);
        double newCoeff = 0;
        if (denominator != 0) {
            newCoeff = numerator / denominator;
        }
        cmlcrf.getWeights().getAllWeights().set(parameterIndex, newCoeff);
    }

//    public void iterate() {
//        // update for each class: 1,...,L
////        System.out.println(cmlcrf.getWeights());
//        for (int l=0; l<numClasses; l++) {
//            iterateForL(l);
//        }
//    }

    /**
     * iterate for all parameters involving class l.
     * e.g. l=1, update w_1^1,...,w_m^1, w_{1,2},...,w_{1,L}
     */
    private void iterateForL(int l) {
        updateClassScoreMatrix();
        updateAssignmentScoreMatrix();
        updateAssignmentProbMatrix();
        updateCombProbSums();
        updateClassProbMatrix();
        //update the gradients and hessians for class l;
        for (int m=-1; m<numFeatures; m++){
            int parameterIndex = l*(numFeatures+1) + m + 1;
            double gradientForFeature = calGradientForFeature(l, m, parameterIndex);
            double hessianForFeature = calHessiansForFeature(l, m);
            double fit = hessianForFeature * cmlcrf.getWeights().getWeightForIndex(parameterIndex) - gradientForFeature;
            double numerator = softThreshold(fit);
            double denominator = hessianForFeature + regularization * (1-l1Ratio);
            double newCoeff = 0;
            if (denominator != 0) {
                newCoeff = numerator / denominator;
            }
            cmlcrf.getWeights().getAllWeights().set(parameterIndex, newCoeff);
        }
        int numLabelPair = numClasses - l - 1;
        if (numLabelPair > 0) {
            for (int l2 = l+1; l2 < numClasses; l2++) {
                for (int pos=labelPairToParams[l][l2]; pos<labelPairToParams[l][l2]+4; pos++) {
                    double gradientForLabelPair = calGradientForLabelPair(pos);
                    double hessiansForLabelPair = calHessiansForLabelPair(pos);
                    int parameterIndex = pos + numWeightsForFeatures;
                    double fit = hessiansForLabelPair * cmlcrf.getWeights().getWeightForIndex(parameterIndex) - gradientForLabelPair;
                    double numerator = softThreshold(fit);
                    double denominator = hessiansForLabelPair + regularization * (1-l1Ratio);
                    double newCoeff = 0;
                    if (denominator != 0) {
                        newCoeff = numerator / denominator;
                    }
                    cmlcrf.getWeights().getAllWeights().set(parameterIndex, newCoeff);
                }
            }
        }
    }

    private static double softThreshold(double z, double gamma){
        if (z>0 && gamma < Math.abs(z)){
            return z-gamma;
        }
        if (z<0 && gamma < Math.abs(z)){
            return z+gamma;
        }
        return 0;
    }

    private double softThreshold(double z){
        return softThreshold(z, regularization*l1Ratio);
    }


    private double calGradientForLabelPair(int pos) {
        double count = 0.0;
        for (int matched: labelPairToCombination.get(pos)){
            count -= combProbSums[matched];
        }

        count += this.empiricalCounts[pos+numWeightsForFeatures];
        return count;
    }

    private double calHessiansForLabelPair(int pos) {
        double count = 0.0;
        for (int matched : labelPairToCombination.get(pos)) {
            count -= combProbSums[matched];
        }

        for (int i=0; i<numData; i++) {
            double[] probs = this.combProbMatrix[i];
            double matchedSum = 0.0;
            for (int matched : labelPairToCombination.get(pos)) {
                matchedSum += probs[matched];
            }
            count += Math.pow(matchedSum,2);
        }
        return count;
    }

    // m: featureIndex
    // l: classIndex
    private double calHessiansForFeature(int l, int m) {
        double count = 0.0;
        if (m == -1) {
            for (int i=0; i<numData; i++) {
                count += (Math.pow(this.classProbMatrix[i][l],2) - this.classProbMatrix[i][l]);
            }
        } else {
            Vector featureColumn = dataSet.getColumn(m);
            for (Vector.Element element : featureColumn.nonZeroes()) {
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count += (Math.pow(this.classProbMatrix[dataPointIndex][l]*featureValue, 2) -
                        this.classProbMatrix[dataPointIndex][l] * Math.pow(featureValue,2));
            }
        }
        return count;
    }

    // m: featureIndex
    // l: classIndex
    private double calGradientForFeature(int l, int m, int parameterIndex) {
        double count = 0.0;
        if (m == -1) {
            for (int i=0; i<numData; i++) {
                count -= this.classProbMatrix[i][l];
            }
        } else {
            Vector featureColumn = dataSet.getColumn(m);
            for (Vector.Element element : featureColumn.nonZeroes()) {
                int dataPointIndex = element.index();
                double featureValue = element.get();
                count -= this.classProbMatrix[dataPointIndex][l] * featureValue;
            }
        }
        count += this.empiricalCounts[parameterIndex];

        return count;
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
        }
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

    private void initCache() {
        parameterToL1 = new int[numWeightsForLabelPairs];
        parameterToL2 = new int[numWeightsForLabelPairs];
        labelPairToParams = new int[numClasses][numClasses];
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
                labelPairToParams[l1][l2] = start;
                start += 4;
            }
        }
        parameterToClass = new int[numWeightsForFeatures];
        parameterToFeature = new int[numWeightsForFeatures];
        for (int i=0; i<numWeightsForFeatures; i++) {
            parameterToClass[i] = cmlcrf.getWeights().getClassIndex(i);
            parameterToFeature[i] = cmlcrf.getWeights().getFeatureIndex(i);
        }

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
    public double getValue() {
        this.value = getValueForAllData() + getPenalty();
        return this.value;
    }

    private double getPenalty() {
        Vector vector = cmlcrf.getWeights().getAllWeights();
        double norm = (1-l1Ratio)*0.5*Math.pow(vector.norm(2),2) + l1Ratio*vector.norm(1);
        return norm * regularization;

    }//check



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
        // score for the true combination
        sum -= combScoreMatrix[i][labelComIndices[i]];
        return sum;
    }


    public Vector getParameters() {
        return cmlcrf.getWeights().getAllWeights();
    }


    private void updateClassScoreMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i-> classScoreMatrix[i] = cmlcrf.predictClassScores(dataSet.getRow(i)));
    }

    private void updateAssignmentScoreMatrix(){
        //todo
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> combScoreMatrix[i] = cmlcrf.predictCombinationScores(classScoreMatrix[i]));
    }

    private void updateAssignmentProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> combProbMatrix[i] = cmlcrf.predictCombinationProbs(combScoreMatrix[i]));
    }

    private void updateClassProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> classProbMatrix[i] = cmlcrf.calClassProbs(combProbMatrix[i]));
    }


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
                case 0:
                    if (!comContainsLabel[c][l1] && !comContainsLabel[c][l2]) list.add(c);
                    break;
                // l1 = 1; l2 = 0;
                case 1: if (comContainsLabel[c][l1] && !comContainsLabel[c][l2]) list.add(c);
                    break;
                // l1 = 0; l2 = 1;
                case 2: if (!comContainsLabel[c][l1] && comContainsLabel[c][l2]) list.add(c);
                    break;
                // l1 = 1; l2 = 1;mapPairToCombination
                case 3: if (comContainsLabel[c][l1] && comContainsLabel[c][l2]) list.add(c);
                    break;
                default: throw new RuntimeException("feature case :" + featureCase + " failed.");
            }
        }
    }

    private void updateCombProbSums(int combinationIndex){
        double sum =0;
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            double prob = combProbMatrix[i][combinationIndex];
            sum += prob;
        }
        combProbSums[combinationIndex] = sum;
    }

    private void updateCombProbSums(){
        IntStream.range(0,numSupport).parallel()
                .forEach(this::updateCombProbSums);
    }


    public Terminator getTerminator() {
        return terminator;
    }
}
