package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Created by Rainicy on 11/27/15.
 */
public class DynamicProgramming {

    public PriorityQueue<Data> dp;

    public double[][] probs;

    public double[][] logProbs;

    /**
     * number of labels;
     */
    public int numLabels;


    /**
     * cache for the Vectors, we had returned so far.
     */
    private Set<Vector> cache;

    /**
     * given probs, format: probs[numLabels][2],
     * two probabilities for each label: prob for 0 and prob for 1.
     * @param probs
     * @param logProbs
     */
    public DynamicProgramming(double[][] probs, double[][] logProbs){
        this.numLabels = probs.length;
        this.probs = probs;
        this.logProbs = logProbs;


        dp = new PriorityQueue<>();
        Vector vector = new DenseVector(numLabels);

        double logProb = 0.0;
        for (int l=0; l<numLabels; l++) {
            if (this.probs[l][1] >= 0.5) {
                vector.set(l, 1.0);
                logProb += this.logProbs[l][1];
            } else {
                logProb += this.logProbs[l][0];
            }
        }
        dp.add(new Data(vector, logProb));
        cache = new HashSet<>();
        cache.add(vector);
    }

    /**
     * calculate the current the highest probability of the
     * first element in the queue.
     * @return
     */
    public double highestProb() {
        if (dp.size() > 0) {
            return Math.exp(dp.peek().logProb);
        }
        return 0;
    }

    /**
     * calculate the current the highest log probability of the
     * first element in the queue.
     * @return
     */
    public double highestLogProb() {
        if (dp.size() > 0) {
            return dp.peek().logProb;
        }
        return Double.NEGATIVE_INFINITY;
    }

    /**
     * find the next vector with highest probability.
     * And update the queue with flipping every label.
     * @return
     */
    public Vector nextHighest() {
        if (dp.size() > 0) {
            flipLabels(dp.peek());
            return dp.poll().vector;
        }

        return new DenseVector(numLabels);
    }

    /**
     * flip each bit in given vector, and calculate its
     * log probability, if it is not cached yet, put it into
     * the max queue.
     * @param data
     */
    private void flipLabels(Data data) {

        double prevlogProb = data.logProb;
        Vector vector = data.vector;

        for (int l=0; l<numLabels; l++) {
            DenseVector flipVector = new DenseVector((DenseVector) vector,false);
            double logProb;
            if (flipVector.get(l) == 0.0) {
                flipVector.set(l, 1.0);
                if (cache.contains(flipVector)) {
                    continue;
                }
                logProb = prevlogProb - this.logProbs[l][0] + this.logProbs[l][1];
            } else {
                flipVector.set(l, 0.0);
                if (cache.contains(flipVector)) {
                    continue;
                }
                logProb = prevlogProb - this.logProbs[l][1] + this.logProbs[l][0];
            }

            dp.add(new Data(flipVector, logProb));
            cache.add(flipVector);
        }
    }

    /**
     * given a vector, return the cluster probability.
     * @param vector
     * @return
     */
    private double calculateProb(DenseVector vector) {
        double logProb = 0.0;
        for (int l=0; l<numLabels; l++) {
            if (vector.get(l) == 1.0) {
                logProb += this.logProbs[l][1];
            } else {
                logProb += this.logProbs[l][0];
            }
        }
        return logProb;
    }

    public String toString() {
        return dp.toString();
    }


    class Data implements Comparable<Data> {
        private final Vector vector;
        private final double logProb;

        Data(Vector vector, double logProb) {
            this.vector = vector;
            this.logProb = logProb;
        }

        @Override
        public int compareTo(Data o) {
            return Double.valueOf(o.logProb).compareTo(logProb);
        }

        public String toString() {
            return "prob: " + String.format("%.3f", Math.exp(logProb)) + "\tvetcor: " + vector;
        }
    }
}

