package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.*;

/**
 * Created by Rainicy on 11/27/15.
 */
public class DynamicProgramming {

    private PriorityQueue<Candidate> queue;

    private double[][] probs;

    private double[][] logProbs;

    /**
     * number of labels;
     */
    private int numLabels;


    /**
     * cache for the Vectors, we had returned so far.
     */
    private Set<MultiLabel> cache;

    /**
     * labels with probabilities between 0 and 1
     */
    private List<Integer> uncertainLabels;

    /**
     *
     * @param probabilities probabilities of heads
     */
    public DynamicProgramming(double[] probabilities){
        double[][] probs = new double[probabilities.length][2];
        double[][] logProbs = new double[probabilities.length][2];
        this.uncertainLabels = new ArrayList<>();
        for (int l=0;l<probabilities.length;l++){
            probs[l][0] = 1-probabilities[l];
            probs[l][1] = probabilities[l];
            logProbs[l][0] = Math.log(probs[l][0]);
            logProbs[l][1] = Math.log(probs[l][1]);
            if (probabilities[l]!=0 && probabilities[l]!=1){
                uncertainLabels.add(l);
            }
        }

        this.numLabels = probs.length;
        this.probs = probs;
        this.logProbs = logProbs;
        cache = new HashSet<>();

        queue = new PriorityQueue<>();
        MultiLabel multiLabel = new MultiLabel();

        double logProb = 0.0;
        for (int l=0; l<numLabels; l++) {
            if (this.probs[l][1] >= 0.5) {
                multiLabel.addLabel(l);
                logProb += this.logProbs[l][1];
            } else {
                logProb += this.logProbs[l][0];
            }
        }
        queue.add(new Candidate(multiLabel, logProb));

        cache.add(multiLabel);
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
        cache = new HashSet<>();

        queue = new PriorityQueue<>();
        MultiLabel multiLabel = new MultiLabel();

        this.uncertainLabels = new ArrayList<>();
        for (int l=0;l<numLabels;l++){
            double p = probs[l][1];
            if (p!=0 && p!=1){
                uncertainLabels.add(l);
            }
        }

        double logProb = 0.0;
        for (int l=0; l<numLabels; l++) {
            if (this.probs[l][1] >= 0.5) {
                multiLabel.addLabel(l);
                logProb += this.logProbs[l][1];
            } else {
                logProb += this.logProbs[l][0];
            }
        }
        queue.add(new Candidate(multiLabel, logProb));

        cache.add(multiLabel);
    }

    public PriorityQueue<Candidate> getQueue() {
        return queue;
    }

    public List<Pair<MultiLabel,Double>> topK(int k){
        List<Pair<MultiLabel,Double>> candidates = new ArrayList<>();
        for (int i=0;i<k;i++){
            Candidate candidate = nextHighest();
            candidates.add(new Pair<>(candidate.multiLabel,candidate.probability));
        }
        return candidates;
    }

    /**
     * calculate the current the highest probability of the
     * first element in the queue.
     * @return
     */
    public double nextHighestProb() {
        if (queue.size() > 0) {
            return queue.peek().probability;
        }
        return 0;
    }

    /**
     * calculate the highest log probability
     * @return
     */
    public double highestLogProb() {
        if (queue.size() > 0) {
            return queue.peek().logProbability;
        }
        return Double.NEGATIVE_INFINITY;
    }

    /**
     * find the next multiLabel with highest probability.
     * And update the queue by flipping every label.
     * @return
     */
    public MultiLabel nextHighestVector() {
        if (queue.size() > 0) {
            flipLabels(queue.peek());
            return queue.poll().multiLabel;
        }

        return new MultiLabel();
    }


    public Candidate nextHighest(){
        if (queue.size() > 0) {
            flipLabels(queue.peek());
            return queue.poll();
        }
        MultiLabel multiLabel = new MultiLabel();
        Candidate candidate = new Candidate(multiLabel, Double.NEGATIVE_INFINITY);
        return candidate;
    }


    /**
     * flip each bit in given multiLabel, and calculate its
     * log probability, if it is not cached yet, put it into
     * the max queue.
     * @param data
     */
    private void flipLabels(Candidate data) {

        double prevlogProb = data.logProbability;
        MultiLabel multiLabel = data.multiLabel;
        // only flip uncertain labels
        for (int l: uncertainLabels) {
            MultiLabel flipped = multiLabel.copy();
            flipped.flipLabel(l);
            double logProb;
            if (flipped.matchClass(l)){
                logProb = prevlogProb - this.logProbs[l][0] + this.logProbs[l][1];
            } else {
                logProb = prevlogProb - this.logProbs[l][1] + this.logProbs[l][0];
            }

            if (!cache.contains(flipped)) {
                queue.add(new Candidate(flipped, logProb));
                cache.add(flipped);
            }

        }
    }

    /**
     * given a multiLabel, return the cluster probability.
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
        return queue.toString();
    }


    public class Candidate implements Comparable<Candidate> {
        private final MultiLabel multiLabel;
        private final double logProbability;
        private final double probability;

        Candidate(MultiLabel multiLabel, double logProbability) {
            this.multiLabel = multiLabel;
            this.logProbability = logProbability;
            this.probability = Math.exp(logProbability);
        }

        public MultiLabel getMultiLabel() {
            return multiLabel;
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
            return "prob: " + String.format("%.3f", Math.exp(logProbability)) + "\tvetcor: " + multiLabel;
        }
    }
}

