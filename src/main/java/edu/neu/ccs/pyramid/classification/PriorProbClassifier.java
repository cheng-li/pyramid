package edu.neu.ccs.pyramid.classification;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

/**
 * Created by chengli on 8/19/14.
 */
public class PriorProbClassifier implements ProbabilityEstimator{
    private static final long serialVersionUID = 1L;

    private int numClasses;
    private double[] probs;
    private int topClass;


    public PriorProbClassifier(int numClasses) {
        this.numClasses = numClasses;
        this.probs = new double[numClasses];
    }

    public void fit(ClfDataSet clfDataSet){
        int[] counts = new int[this.numClasses];
        int[] labels = clfDataSet.getLabels();
        for (int label: labels){
            counts[label] += 1;
        }
        int numDataPoints = clfDataSet.getNumDataPoints();
        for (int k=0;k<this.numClasses;k++){
            this.probs[k] = (double)counts[k]/numDataPoints;
        }

        this.topClass = IntStream.range(0, this.numClasses)
                .mapToObj(k -> new Pair<>(k,counts[k]))
                .max(Comparator.comparing(Pair::getSecond))
                .get().getFirst();
    }

    @Override
    public int predict(Vector vector) {
        return this.topClass;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    public double[] predictClassProbs(Vector vector) {
        return this.probs;
    }

    public double[] getClassProbs(){
        return this.probs;
    }

    @Override
    public String toString() {
        return "PriorProbClassifier{" +
                "numClasses=" + numClasses +
                ", probs=" + Arrays.toString(probs) +
                ", topClass=" + topClass +
                '}';
    }
}
