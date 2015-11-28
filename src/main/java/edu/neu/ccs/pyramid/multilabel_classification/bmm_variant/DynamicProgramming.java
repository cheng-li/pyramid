package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import java.util.PriorityQueue;

/**
 * Created by Rainicy on 11/27/15.
 */
public class DynamicProgramming {

    public PriorityQueue<Data> dp;

    public double[] probs;

    /**
     * number of labels;
     */
    public int numLabels;

    /**
     * given probs, format: probs[numLabels][2],
     * two probabilities for each label: prob for 0 and prob for 1.
     * @param probsForTwo
     */
    public DynamicProgramming(double[][] probsForTwo){
        this.numLabels = probs.length;
        this.probs = new double[numLabels];
        for (int l=0; l<numLabels; l++) {
            this.probs[l] = probsForTwo[l][1];
        }

        dp = new PriorityQueue<>();

        Vector vector = new DenseVector(numLabels);

        double prob = 1.0;
        for (int l=0; l<numLabels; l++) {
            if (this.probs[l] >= 0.5) {
                vector.set(l, 1.0);
                prob *= this.probs[l];
            } else {
                prob *= (1 - this.probs[l]);
            }
        }
        dp.add(new Data(vector, prob));
    }

    //TODO: 
    public Vector getNextHighest() {

    }


    class Data implements Comparable<Data> {
        private final Vector vector;
        private final double prob;

        Data(Vector vector, double prob) {
            this.vector = vector;
            this.prob = prob;
        }

        @Override
        public int compareTo(Data o) {
            return Double.valueOf(o.prob).compareTo(prob);
        }
    }
}

