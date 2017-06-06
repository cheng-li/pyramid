package edu.neu.ccs.pyramid.core.multilabel_classification.plugin_rule;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.core.dataset.MultiLabel;
import edu.neu.ccs.pyramid.core.eval.InstanceAverage;
import edu.neu.ccs.pyramid.core.multilabel_classification.Enumerator;
import edu.neu.ccs.pyramid.core.eval.KLDivergence;
import edu.neu.ccs.pyramid.core.util.Pair;
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

    /**
     *
     * @param numClasses
     * @param multiLabels combinations with non-zero probabilities
     * @param probabilities associated probabilities
     * @return
     */
    public static MultiLabel predict(int numClasses, List<MultiLabel> multiLabels, List<Double> probabilities){
        Matrix p = getPMatrix(numClasses, multiLabels, probabilities);
        Matrix delta = getDeltaMatrix(p);
        double zeroProb = 0;
        for (int i=0;i<multiLabels.size();i++){
            if (multiLabels.get(i).getMatchedLabels().size()==0){
                zeroProb = probabilities.get(i);
                break;
            }
        }
        return predictByDeltaMatrix(delta,zeroProb).getFirst();
    }

    public static MultiLabel predict(int numClasses, List<MultiLabel> multiLabels, double[] probabilities){
        List<Double> p = Arrays.stream(probabilities).mapToObj(a->a).collect(Collectors.toList());
        return predict(numClasses,multiLabels,p);
    }


    /**
     *
     * @param numClasses
     * @param samples sampled multi-labels; can have duplicates; their empirical probabilities will be estimated
     * @return
     */
    public static MultiLabel predict(int numClasses, List<MultiLabel> samples){
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

    /**
     *
     * @param deltaMatrix
     * @param zeroProbability
     * @return best multi-label and F1
     */
    private static Pair<MultiLabel,Double> predictByDeltaMatrix(Matrix deltaMatrix, double zeroProbability){
        int numClasses = deltaMatrix.numCols();
        MultiLabel pred = null;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int k=1;k<=numClasses;k++){
            List<Pair<Integer,Double>> column = new ArrayList<>();

            for (int i=1;i<=numClasses;i++){
                column.add(new Pair<>(i-1,deltaMatrix.get(i-1,k-1)));
            }

            Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(Pair::getSecond);
            List<Pair<Integer,Double>> sorted = column.stream().sorted(comparator.reversed())
                    .collect(Collectors.toList());
            MultiLabel multiLabel = new MultiLabel();
            double value = 0;
            for (int l=0;l<k;l++){
                multiLabel.addLabel(sorted.get(l).getFirst());
                value += sorted.get(l).getSecond();
            }

            if (value > maxValue){
                maxValue = value;
                pred = multiLabel;
            }
        }

        if (zeroProbability > maxValue){
            pred = new MultiLabel();
            maxValue = zeroProbability;
        }

        return new Pair<>(pred,maxValue);
    }

    private static Matrix getDeltaMatrix(Matrix pMatrix){
        int size = pMatrix.numRows();
        DenseMatrix wMatrix = new DenseMatrix(size,size);
        for (int s=1;s<=size;s++){
            for (int k=1;k<=size;k++){
                wMatrix.set(s-1,k-1,2.0/(s+k));
            }
        }
        return pMatrix.times(wMatrix);
    }


    /**
     *
     * @param pMatrix access: matrix[l][s-1] = score for label l (0~L-1), size s (1~L)
     * @return
     */
    public static MultiLabel predict(Matrix pMatrix, double zeroProbability){
        Matrix deltaMatrix = getDeltaMatrix(pMatrix);
        return predictByDeltaMatrix(deltaMatrix,zeroProbability).getFirst();
    }

    private static Matrix getPMatrix(int numClasses, List<MultiLabel> multiLabels, List<Double> probabilities){
        DenseMatrix pMatrix = new DenseMatrix(numClasses,numClasses);
        for (int j=0;j<multiLabels.size();j++){
            MultiLabel multiLabel = multiLabels.get(j);
            double prob = probabilities.get(j);
            int s = multiLabel.getMatchedLabels().size();
            for (int i: multiLabel.getMatchedLabels()){
                double old = pMatrix.get(i,s-1);
                pMatrix.set(i,s-1,old+prob);
            }
        }
        return pMatrix;
    }


    public static Matrix getPMatrix(int numClasses, List<MultiLabel> samples){
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
        Matrix p = getPMatrix(numClasses, uniqueOnes, probs);
        return p;
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

    public static Matrix getTruePMatrix(int numClasses, MultiLabel trueMultiLabel){
        int s = trueMultiLabel.getNumMatchedLabels();
        DenseMatrix pMatrix = new DenseMatrix(numClasses,numClasses);
        for (int l: trueMultiLabel.getMatchedLabels()){
            pMatrix.set(l,s-1,1);
        }
        return  pMatrix;
    }

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
