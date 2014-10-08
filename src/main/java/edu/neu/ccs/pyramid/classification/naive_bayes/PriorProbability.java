package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;

import java.util.Arrays;
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
    protected double[] priors;

    private int numClasses;

    /** Constructor by given number of classes. */
    public PriorProbability(int numClasses) throws IllegalArgumentException {
        if (numClasses <= 0) {
            throw new IllegalArgumentException("Number of " +
                    "classes cannot be a negtive");
        }
        this.numClasses = numClasses;
        priors = new double[numClasses];
    }

    /** Constructor by given number of classes and labels */
    public PriorProbability(int numClasses, int[] labels)
            throws IllegalArgumentException
    {
        this(numClasses);
        setPriors(labels);
    }

    /** Constructor by given classic data set */
    public PriorProbability(ClfDataSet clfDataSet) {
        this(clfDataSet.getNumClasses(), clfDataSet.getLabels());
    }


    /** Getter the number types of labels. */
    public int getNumClasses(){
        return this.numClasses;
    }

    /** Getter the prior probabilities. */
    public double[] getPriors() {
        return this.priors;
    }

    /** Given by the labels, fitting the priors. */
    public void setPriors(int[] labels) {
        if (labels.length  == 0) {
            throw new IllegalArgumentException("Given labels' " +
                    "length equals zero.");
        }

        // calculate the prior probabilities.
        int[] counts = new int[numClasses];
        for (int i=0; i<labels.length; i++) {
            counts[labels[i]] += 1;
        }

        for ( int i=0; i<numClasses; i++ ) {
            priors[i] = (double)counts[i] / labels.length;
        }
    }

    /** Getter the prior probability by given label. */
    public double getPriorProb(int label) throws IllegalArgumentException {
        checkLabel(label);
        return priors[label];
    }

    /** Getter the log prior probability by given label. */
    public double logPriorProb(int label) throws IllegalArgumentException {
        checkLabel(label);
        return Math.log(priors[label]);
    }

    /** Check whether the given label is illegal */
    protected void checkLabel(int label) throws IllegalArgumentException {
        if ((label<0) || (label>=numClasses))  {
            throw new IllegalArgumentException("Label does not exist.");
        }
    }


    public String toString() {
        String str = "Prior Probability Class {" +
                "number of classes = " + getNumClasses() +
                ", priors = [";
        for (int i=0; i<numClasses; i++) {
            str += i + ": " + priors[i] + "; ";
        }
        str += "]}";
        return str;
    }

    @Override
    public boolean isValid() {
        double sum = 0;
        for (int i=0; i<numClasses; i++) {
            double prob = priors[i];
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
