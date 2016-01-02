package edu.neu.ccs.pyramid.multilabel_classification;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Created by Rainicy on 11/27/15.
 */
public class DynamicProgramming {

    public PriorityQueue<Candidate> dp;

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
     *
     * @param probabilities probabilities of heads
     */
    public DynamicProgramming(double[] probabilities){
        double[][] probs = new double[probabilities.length][2];
        double[][] logProbs = new double[probabilities.length][2];
        for (int l=0;l<probabilities.length;l++){
            probs[l][0] = 1-probabilities[l];
            probs[l][1] = probabilities[l];
            logProbs[l][0] = Math.log(probs[l][0]);
            logProbs[l][1] = Math.log(probs[l][1]);
        }

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
        dp.add(new Candidate(vector, logProb));
        cache = new HashSet<>();
        cache.add(vector);
    }

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
        dp.add(new Candidate(vector, logProb));
        cache = new HashSet<>();
        cache.add(vector);
    }

    /**
     * calculate the current the highest probability of the
     * first element in the queue.
     * @return
     */
    public double nextHighestProb() {
        if (dp.size() > 0) {
            return Math.exp(dp.peek().logProbability);
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
            return dp.peek().logProbability;
        }
        return Double.NEGATIVE_INFINITY;
    }

    /**
     * find the next vector with highest probability.
     * And update the queue with flipping every label.
     * @return
     */
    public Vector nextHighestVector() {
        if (dp.size() > 0) {
            flipLabels(dp.peek());
            return dp.poll().vector;
        }

        return new DenseVector(numLabels);
    }


    public Candidate nextHighest(){
        if (dp.size() > 0) {
            flipLabels(dp.peek());
            return dp.poll();
        }
        Vector vector = new DenseVector(numLabels);
        Candidate candidate = new Candidate(vector, Double.NEGATIVE_INFINITY);
        return candidate;
    }


    /**
     * flip each bit in given vector, and calculate its
     * log probability, if it is not cached yet, put it into
     * the max queue.
     * @param data
     */
    private void flipLabels(Candidate data) {

        double prevlogProb = data.logProbability;
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

            dp.add(new Candidate(flipVector, logProb));
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


    public class Candidate implements Comparable<Candidate> {
        private final Vector vector;
        private final double logProbability;
        private final double probability;

        Candidate(Vector vector, double logProbability) {
            this.vector = vector;
            this.logProbability = logProbability;
            this.probability = Math.exp(logProbability);
        }

        public Vector getVector() {
            return vector;
        }

        public double getLogProbability() {
            return logProbability;
        }

        public double getProbability() {
            return probability;
        }

        @Override
        public int compareTo(Candidate o) {
            return Double.valueOf(o.logProbability).compareTo(logProbability);
        }

        public String toString() {
            return "prob: " + String.format("%.3f", Math.exp(logProbability)) + "\tvetcor: " + vector;
        }
    }
}

