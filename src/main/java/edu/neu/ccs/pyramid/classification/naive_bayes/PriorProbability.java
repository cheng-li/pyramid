package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Rainicy on 10/4/14.
 *
 * A class contains prior probabilities corresponding
 * to the labels/categories.
 *
 */
public class PriorProbability implements Probability {

    /** Prior probabilities map. */
    protected Map<Integer, Double> priors;

    private int numClasses;

    /** Default constructor */
    public PriorProbability() {
        priors = new HashMap<Integer, Double>();
        this.numClasses = 0;
    }

    /** Constructor by given labels
     * @param labels*/
    public PriorProbability(int[] labels) {
        priors = new HashMap<Integer, Double>();

        if (labels.length  == 0) {
            throw new IllegalArgumentException("Given labels' " +
                    "length equals zero.");
        }
        // calculate the prior probabilities.
        for ( Integer label : labels ) {
            if (!priors.containsKey(label)) {
                priors.put(label, new Double(1));
            }
            else {
                priors.put(label, priors.get(label) + 1);
            }
        }

        for ( Integer label : priors.keySet() ) {
            priors.put(label, (double)priors.get(label) / labels.length);
        }

        this.numClasses = priors.size();
    }

    /** Constructor by given classic data set */
    public PriorProbability(ClfDataSet clfDataSet) {
        this(clfDataSet.getLabels());
    }


    /** Getter the number types of labels. */
    public int getNumClasses(){
        return this.numClasses;
    }

    /** Getter the prior probabilities. */
    public Map getPriors() {
        return this.priors;
    }

    /** Getter the prior probability by given label. */
    public double getPriorProb(Integer label) throws IllegalArgumentException {
        checkLabel(label);
        return priors.get(label);
    }

    /** Getter the log prior probability by given label. */
    public double logPriorProb(Integer label) throws IllegalArgumentException {
        checkLabel(label);
        return Math.log(priors.get(label));
    }

    /** Check whether the given label is illegal */
    protected void checkLabel(Integer label) throws IllegalArgumentException {
        if (!priors.containsKey(label)) {
            throw new IllegalArgumentException("Label does not exist.");
        }
    }


    public String toString() {
        return "Prior Probability Class {" +
                "number of classes = " + getNumClasses() +
                ", priors = " + priors.values() + "}";
    }

    @Override
    public boolean isValid() {
        double sum = 0;
        for (Integer label : priors.keySet()) {
            double prob = priors.get(label);
            if (prob < 0 || prob > 1) {
                return false;
            }
            sum += prob;
        }
        if (Math.abs(1-sum) > THRESHOLD) {
            return false;
        }

        return true;
    }

    /**
     * TODO: Add a prior probability by given label and prior.
     */
}
