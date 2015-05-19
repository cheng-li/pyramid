package edu.neu.ccs.pyramid.classification.naive_bayes;

import org.apache.commons.math3.util.FastMath;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Rainicy on 10/10/14.
 *
 * Multinomial Distribution.
 *
 * @see edu.neu.ccs.pyramid.classification.naive_bayes.Distribution
 *
 * @reference
 * http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
 */
public class Multinomial implements Distribution {

    /** Phis to save ratios of each value to
     *  total number counts. *
     */
    private Map<Integer, Double> phis;

    /** default log density by given variable is 0. */
    private double defaultZeroLogProb;

    /**
     * If we cannot find the given frequency in
     * the dictionary phis, returns missingPhi.
     */
    private double missingPhi;


    /** Default constructor. */
    public Multinomial(Map<Integer, Double> phis) {
        this.phis = phis;
    }

    /** Constructor by given variables. */
    public Multinomial(double[] nonzeroVars, int numPerClass) {
        fit(nonzeroVars, numPerClass);
    }

    @Override
    public void fit(double[] nonzeroVars, int numPerClass) throws IllegalArgumentException {

        Map<Integer, Integer> frequencies = new HashMap<>();
        // first put key=0.
        frequencies.put(0, numPerClass-nonzeroVars.length);
        for (int i=0; i<nonzeroVars.length; i++) {
            Integer key = new Integer((int) nonzeroVars[i]);
            if (frequencies.containsKey(key)) {
                frequencies.put(key, frequencies.get(key)+1);
            }
            else {
                frequencies.put(key, 1);
            }
        }

        this.phis = new HashMap<>();
        double sumProbability = 0;
        int totalKeySize = frequencies.size();
//        int totalSize = variables.length;
        for (Integer integer : frequencies.keySet()) {
            double phi = 0;
            phi = ((double) frequencies.get(integer) + 1) / (numPerClass+totalKeySize+1);
            phis.put(integer, phi);
            sumProbability += phi;
        }
        this.missingPhi = 1 - sumProbability;
        this.defaultZeroLogProb = logProbability(0);
    }

    @Override
    public double probability(double x) throws IllegalArgumentException {
        return FastMath.exp(logProbability(x));
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        if (x == 0.0) {
            return this.defaultZeroLogProb;
        }
        Integer integer = new Integer((int) x);
        if (this.phis.containsKey(integer)) {
            return Math.log(phis.get(integer));
        }
        return Math.log(missingPhi);
    }
}
