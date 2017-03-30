package edu.neu.ccs.pyramid.multilabel_classification.plugin_rule;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.eval.InstanceAverage;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.util.ArgSort;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.multilabel_classification.Enumerator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Based on the paper
 * On the bayes-optimality of f-measure maximizers.
 * The Journal of Machine Learning Research, 15(1):3333–3388, 2014.
 * Section 5.
 * Created by chengli on 4/5/16.
 */
public class GeneralF1Predictor {
    // max label set size
    //todo set
    private int maxSize = 15;

    /**
     *
     * @param numClasses
     * @param multiLabels combinations with non-zero probabilities
     * @param probabilities associated probabilities
     * @return
     */
    public MultiLabel predict(int numClasses, List<MultiLabel> multiLabels, List<Double> probabilities){
        double[][] p = getPMatrix(numClasses, multiLabels, probabilities);
        double zeroProb = 0;
        for (int i=0;i<multiLabels.size();i++){
            if (multiLabels.get(i).getMatchedLabels().size()==0){
                zeroProb = probabilities.get(i);
                break;
            }
        }
        return predictWithPMatrix(p, zeroProb);
    }

    public MultiLabel predict(int numClasses, List<MultiLabel> multiLabels, double[] probabilities){
        List<Double> p = Arrays.stream(probabilities).mapToObj(a->a).collect(Collectors.toList());
        return predict(numClasses,multiLabels,p);
    }


    /**
     *
     * @param numClasses
     * @param samples sampled multi-labels; can have duplicates; their empirical probabilities will be estimated
     * @return
     */
    public MultiLabel predict(int numClasses, List<MultiLabel> samples){
        Multiset<MultiLabel> multiset = ConcurrentHashMultiset.create();
        for (MultiLabel multiLabel: samples){
            multiset.add(multiLabel);
        }

        int sampleSize = samples.size();
        List<MultiLabel> uniqueOnes = new ArrayList<>();
        List<Double> probs = new ArrayList<>();
        for (Multiset.Entry<MultiLabel> entry: multiset.entrySet()){
            uniqueOnes.add(entry.getElement());
            probs.add((double)entry.getCount()/sampleSize);
        }
        return predict(numClasses,uniqueOnes,probs);
    }



    public MultiLabel predictWithPMatrix(double[][] pMatrix, double zeroProbability){
        int numLabels = pMatrix.length;
        int min = Math.min(maxSize, numLabels);
        MultiLabel best = new MultiLabel();
        double bestScore = zeroProbability;
        for (int k=1;k<=min;k++){
            double[] deltaVector = getDeltaVector(pMatrix, k);
            Pair<MultiLabel, Double> innerBest = bestWithLengthK(deltaVector, k);
            if (innerBest.getSecond()>bestScore){
                bestScore = innerBest.getSecond();
                best = innerBest.getFirst();
            }
        }
        return best;
    }

    private Pair<MultiLabel, Double> bestWithLengthK(double[] deltaVector, int k){
        int[] sortedIndcies = ArgSort.argSortDescending(deltaVector);
        MultiLabel multiLabel = new MultiLabel();
        double score = 0;
        for (int i=0;i<k;i++){
            int label = sortedIndcies[i];
            multiLabel.addLabel(label);
            score += deltaVector[label];
        }
        return new Pair<>(multiLabel, score);
    }

    /**
     *
     * @param pMatrix
     * @param size k
     * @return delta_ik i=0,1...,L for a fixed k
     */
    private double[] getDeltaVector(double[][] pMatrix, int size){
        int numLabels = pMatrix.length;
        int min = Math.min(maxSize, numLabels);
        double[] d = new double[numLabels];
        for (int i=0;i<numLabels;i++){
            double sum = 0;
            for (int s=1;s<=min;s++){
                sum += 2*pMatrix[i][s-1]/(s+size);
            }
            d[i] = sum;
        }
        return d;
    }


    private double[][] getPMatrix(int numClasses, List<MultiLabel> multiLabels, List<Double> probabilities){
        int min = Math.min(maxSize, numClasses);
        double[][] pMatrix = new double[numClasses][min];
        for (int j=0;j<multiLabels.size();j++){
            MultiLabel multiLabel = multiLabels.get(j);
            double prob = probabilities.get(j);
            int s = multiLabel.getMatchedLabels().size();
            if (s<=maxSize){
                for (int i: multiLabel.getMatchedLabels()){
                    double old = pMatrix[i][s-1];
                    pMatrix[i][s-1]=old+prob;
                }
            }

        }
        return pMatrix;
    }


    public static MultiLabel exhaustiveSearch(int numClasses, Matrix lossMatrix, List<Double> probabilities){
        double bestScore = Double.POSITIVE_INFINITY;
        Vector vector = new DenseVector(probabilities.size());
        for (int i=0;i<vector.size();i++){
            vector.set(i,probabilities.get(i));
        }
        List<MultiLabel> multiLabels = Enumerator.enumerate(numClasses);
        MultiLabel multiLabel = null;
        for (int j=0;j<lossMatrix.numCols();j++){
            Vector column = lossMatrix.viewColumn(j);
            double score = column.dot(vector);
            System.out.println("column "+j+", expected loss = "+score);
            if (score < bestScore){
                bestScore = score;
                multiLabel = multiLabels.get(j);
            }
        }
        return multiLabel;
    }
//
//    public static Matrix getTruePMatrix(int numClasses, MultiLabel trueMultiLabel){
//        int s = trueMultiLabel.getNumMatchedLabels();
//        DenseMatrix pMatrix = new DenseMatrix(numClasses,numClasses);
//        for (int l: trueMultiLabel.getMatchedLabels()){
//            pMatrix.set(l,s-1,1);
//        }
//        return  pMatrix;
//    }

    /**
     * the expected F1 of the target combination under the estimated joint
     * @param combinations
     * @param probs
     * @param target
     * @param numClasses
     * @return
     */
    public static double expectedF1(List<MultiLabel> combinations, double[] probs, MultiLabel target, int numClasses){
        double sum = 0;
        for (int i=0;i<combinations.size();i++){
            sum += probs[i]*(new InstanceAverage(numClasses,combinations.get(i),target).getF1());
        }
        return sum;
    }

    public static Analysis showSupportPrediction(List<MultiLabel> combinations, double[] probs, MultiLabel truth, MultiLabel prediction, int numClasses){
        int truthIndex = 0;
        for (int i=0;i<combinations.size();i++){
            if (combinations.get(i).equals(truth)){
                truthIndex = i;
                break;
            }
        }

        double[] trueJoint = new double[combinations.size()];
        trueJoint[truthIndex] = 1;
        double kl = KLDivergence.kl(trueJoint, probs);

        List<Pair<MultiLabel, Double>> list = new ArrayList<>();
        for (int i=0;i<combinations.size();i++){
            list.add(new Pair<>(combinations.get(i),probs[i]));
        }
        Comparator<Pair<MultiLabel, Double>> comparator = Comparator.comparing(a-> a.getSecond());
        List<Pair<MultiLabel, Double>> sorted = list.stream().sorted(comparator.reversed()).filter(pair->pair.getSecond()>0.01).collect(Collectors.toList());

        double expectedF1Prediction = expectedF1(combinations, probs, prediction, numClasses);

        double expectedF1Truth = expectedF1(combinations, probs, truth, numClasses);

        double actualF1 = new InstanceAverage(numClasses, truth, prediction).getF1();

        StringBuilder jointString = new StringBuilder();
        for (int i=0;i<sorted.size();i++){
            jointString.append(sorted.get(i).getFirst()).append(":").append(sorted.get(i).getSecond()).append(", ");
        }

        Analysis analysis = new Analysis();
        analysis.expectedF1Prediction = expectedF1Prediction;
        analysis.expectedF1Truth = expectedF1Truth;
        analysis.actualF1 = actualF1;
        analysis.kl = kl;
        analysis.prediction = prediction;
        analysis.truth = truth;
        analysis.joint = jointString.toString();

        return analysis;
    }

    public static class Analysis{
        double expectedF1Prediction;
        double expectedF1Truth;
        double actualF1;
        double kl;
        MultiLabel truth;
        MultiLabel prediction;
        String joint;

        public double getExpectedF1Prediction() {
            return expectedF1Prediction;
        }

        public double getExpectedF1Truth() {
            return expectedF1Truth;
        }

        public double getActualF1() {
            return actualF1;
        }

        public double getKl() {
            return kl;
        }

        public MultiLabel getTruth() {
            return truth;
        }

        public MultiLabel getPrediction() {
            return prediction;
        }

        public String getJoint() {
            return joint;
        }

        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder();
            sb.append("truth=").append(truth).append("\n");
            sb.append("prediction=").append(prediction).append("\n");
            sb.append("actual F1=").append(actualF1).append("\n");
            sb.append("kl=").append(kl).append("\n");
            sb.append("expected F1 of truth=").append(expectedF1Truth).append("\n");
            sb.append("expected F1 of prediction=").append(expectedF1Prediction).append("\n");
            sb.append("joint=").append(joint).append("\n");
            return sb.toString();
        }
    }
}
