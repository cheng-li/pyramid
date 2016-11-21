package edu.neu.ccs.pyramid.multilabel_classification.crf;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.dataset.SequentialSparseDataSet;
import edu.neu.ccs.pyramid.optimization.Terminator;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 11/10/16.
 *
 * Train CMLCRF using ElasticNet by coordinate Descent
 */
public class CMLCRFElasticNet {

    private Terminator terminator;
    private CMLCRF cmlcrf;
    private List<MultiLabel> supportedCombinations;
    private int numSupport;
    private MultiLabelClfDataSet dataSet;
    private int numClasses;
    private int numParameters;
    private int numWeightsForFeatures;
    private int numWeightsForLabelPairs;
    private double value;
    private int[] parameterToL1;
    private int[] parameterToL2;
    private int[] parameterToClass;
    private int[] parameterToFeature;
    // whether the support combination contains the label;
    // size num combination* num classes
    private boolean[][] comContainsLabel;
    private boolean isParallel = true;
    private boolean isValueCacheValid = false;

    // numDataPoints by numClasses;
    private double[][] classScoreMatrix;

    // numDataPoints by numCombinations
    private double[][] combProbMatrix;

    // numDataPoints by numCombinations
    private double[][] combScoreMatrix;

    private int numData;
    // for each data point, store the position of the true combination in the support list
    private int[] labelComIndices;

    private double l1Ratio;
    private double regularization;
    private int numFeature;
    // label-lable features
    // size = numSupport * (for each support, label-label feature is non-zero index, starting from 0)
    private List<List<Integer>> combinationToLabelPair;

    public CMLCRFElasticNet (CMLCRF cmlcrf, MultiLabelClfDataSet dataSet, double l1Ratio, double regularization) {
        this.l1Ratio = l1Ratio;
        this.regularization = regularization;
        this.terminator = new Terminator();
        this.terminator.setGoal(Terminator.Goal.MINIMIZE);
        this.numFeature = dataSet.getNumFeatures();

        this.cmlcrf = cmlcrf;
        this.supportedCombinations = cmlcrf.getSupportCombinations();
        this.numSupport = cmlcrf.getNumSupports();
        this.dataSet = dataSet;
        this.numData = dataSet.getNumDataPoints();
        this.numClasses = dataSet.getNumClasses();
        this.numParameters = cmlcrf.getWeights().totalSize();
        this.numWeightsForFeatures = cmlcrf.getWeights().getNumWeightsForFeatures();
        this.numWeightsForLabelPairs = cmlcrf.getWeights().getNumWeightsForLabels();
        this.classScoreMatrix = new double[numData][numClasses];
        this.combScoreMatrix = new double[numData][numSupport];
        this.combProbMatrix = new double[numData][numSupport];
        this.isValueCacheValid = false;
        this.initCache();
        this.combinationToLabelPair = new ArrayList<>(numSupport);
        for (int i=0;i< numSupport;i++) {
            combinationToLabelPair.add(new LinkedList<>());
        }
        this.mapCombinattionToPair();


        Map<MultiLabel,Integer> map = new HashMap<>();
        for (int s=0;s< numSupport;s++){
            map.put(supportedCombinations.get(s),s);
        }
        this.labelComIndices = new int[dataSet.getNumDataPoints()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            labelComIndices[i] = map.get(dataSet.getMultiLabels()[i]);
        }
    }


    public void optimize() {
        while (true) {
            iterate();
            if (terminator.shouldTerminate()) {
                break;
            }
        }
    }

    public void iterate() {
//        System.out.println("weights: " + cmlcrf.getWeights().getAllWeights());
        // O(NdL)
//        System.out.println(Arrays.toString(cmlcrf.getCombinationLabelPartScores()));
        updateClassScoreMatrix();
        cmlcrf.updateCombLabelPartScores();
        updateAssignmentScoreMatrix();
        updateAssignmentProbMatrix();
        // update for each support label set
        for (int l=0; l<numSupport; l++) {
//            System.out.println("label: " + supportedCombinations.get(l));
            DataSet newData = expandData(l);
            iterateForOneComb(newData, l);
        }
        this.terminator.add(getValue());
    }

    private DataSet expandData(int l) {
        SequentialSparseDataSet newData = new SequentialSparseDataSet(numData, numParameters, false);
        MultiLabel label = supportedCombinations.get(l);
        List<Integer> labelPairForL = combinationToLabelPair.get(l);
        // TODO: parallelism
        for (int i=0; i<numData; i++) {
            // add feature-label feature
            for (int y : label.getMatchedLabels()) {
                // set bias as 1
                newData.setFeatureValue(i, (numFeature+1)*y, 1.0);
                for (Vector.Element element : dataSet.getRow(i).nonZeroes()) {
                    int index = element.index();
                    double value = element.get();
                    newData.setFeatureValue(i, (numFeature+1)*y+index+1, value);
                }
            }
            for (int y : labelPairForL) {
                newData.setFeatureValue(i, (numWeightsForFeatures+y), 1.0);
            }
        }
        return newData;
    }

    // update
    private void iterateForOneComb(DataSet newData, int l) {
        double[] realLabels = new double[numData];
        double[] instanceWeights = new double[numData];
        IntStream.range(0, numData).parallel().forEach(i -> {
            double prob = combProbMatrix[i][l];
            double classScore = combScoreMatrix[i][l];
            int y = labelComIndices[i];

            double frac = 0;
            double tmpP = prob * (1-prob);
            int indicator = (y==l)?1:0;
            if (prob!=0&&prob!=1) {
                frac = (indicator-prob) / tmpP;
            }

            if (frac>1) {
                frac=1;
            }
            if (frac<-1) {
                frac=-1;
            }
            realLabels[i] = classScore + frac;
            instanceWeights[i] = tmpP;
        });

        CRFLinearRegression linearRegression = new CRFLinearRegression(numParameters,cmlcrf.getWeights().getAllWeights());
        CRFElasticNetLinearRegOptimizer linearRegTrainer = new CRFElasticNetLinearRegOptimizer(linearRegression, newData, realLabels, instanceWeights);
        linearRegTrainer.setRegularization(regularization);
        linearRegTrainer.setL1Ratio(l1Ratio);
        linearRegTrainer.optimize();
        isValueCacheValid = false;
    }



    private void updateClassScoreMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i-> classScoreMatrix[i] = cmlcrf.predictClassScores(dataSet.getRow(i)));
    }

    private void updateAssignmentScoreMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> combScoreMatrix[i] = cmlcrf.predictCombinationScores(classScoreMatrix[i]));
    }

    private void updateAssignmentProbMatrix(){
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i -> combProbMatrix[i] = cmlcrf.predictCombinationProbs(combScoreMatrix[i]));
    }

    private void initCache() {
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

        comContainsLabel = new boolean[numSupport][numClasses];
        for (int num=0; num< numSupport; num++) {
            for (int l=0; l<numClasses; l++) {
                if (supportedCombinations.get(num).matchClass(l)) {
                    comContainsLabel[num][l] = true;
                }
            }
        }

    }

    private void mapCombinattionToPair() {
        IntStream.range(0, numSupport).forEach(this::mapCombinattionToPair);
    }

    private void mapCombinattionToPair(int s) {
        for (int position=0; position<numWeightsForLabelPairs; position++){
            int l1 = parameterToL1[position];
            int l2 = parameterToL2[position];
            int featureCase = position % 4;
            switch (featureCase) {
                // both l1, l2 equal 0;
                case 0: if (!comContainsLabel[s][l1] && !comContainsLabel[s][l2]) combinationToLabelPair.get(s).add(position);
                    break;
                // l1 = 1; l2 = 0;
                case 1: if (comContainsLabel[s][l1] && !comContainsLabel[s][l2]) combinationToLabelPair.get(s).add(position);
                    break;
                // l1 = 0; l2 = 1;
                case 2: if (!comContainsLabel[s][l1] && comContainsLabel[s][l2]) combinationToLabelPair.get(s).add(position);
                    break;
                // l1 = 1; l2 = 1;
                case 3: if (comContainsLabel[s][l1] && comContainsLabel[s][l2]) combinationToLabelPair.get(s).add(position);
                    break;
                default: throw new RuntimeException("feature case :" + featureCase + " failed.");
            }
        }

    }




    /**
     * @return negative log-likelihood
     */
    public double getValue() {
        if (isValueCacheValid) {
            return this.value;
        }

        this.value = getValueForAllData() + getPenalty();
        this.isValueCacheValid = true;
        return this.value;
    } //check

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
    }//check
    // NLL
    private double getValueForOneData(int i) {
        double sum = 0.0;
        // sum logZ(x_n)
        sum += MathUtil.logSumExp(combScoreMatrix[i]);
        // score for the true combination
        sum -= combScoreMatrix[i][labelComIndices[i]];
        return sum;
    }//check

}
