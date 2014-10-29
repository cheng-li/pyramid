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
    public Multinomial(double[] variables) {
        fit(variables);
    }

    @Override
    public void fit(double[] variables) throws IllegalArgumentException {
        if (variables.length <= 0) {
            throw new IllegalArgumentException("Given variables should " +
                    "be more than 0.");
        }

        Map<Integer, Integer> frequencies = new HashMap<>();
        for (int i=0; i<variables.length; i++) {
            Integer key = new Integer((int) variables[i]);
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
        int totalSize = variables.length;
        for (Integer integer : frequencies.keySet()) {
            double phi = 0;
            phi = ((double) frequencies.get(integer) + 1) / (totalSize+totalKeySize+1);
            phis.put(integer, phi);
            sumProbability += phi;
        }
        this.missingPhi = 1 - sumProbability;
    }

    @Override
    public double probability(double x) throws IllegalArgumentException {
        Integer integer = new Integer((int) x);
        if (this.phis.containsKey(integer)) {
            return phis.get(integer);
        }
        return missingPhi;
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        return FastMath.log(probability(x));
    }

    @Override
    public double cumulativeProbability(double x) throws IllegalAccessException {
        throw new IllegalAccessException("Mutinomial distribution cannot " +
                "support cumulative operation.");
    }

    /**
     * Because of smoothing, cannot get mean easily.
     * @return
     * @throws IllegalAccessException
     */
    @Override
    public double getMean() throws IllegalAccessException {
        throw new IllegalAccessException("Mutinomial distribution cannot " +
                "support getMean operation.");
    }

    /**
     * Because of smoothing, cannot get variance easily.
     * @return
     * @throws IllegalAccessException
     */
    @Override
    public double getVariance() throws IllegalAccessException {
        throw new IllegalAccessException("Mutinomial distribution cannot " +
                "support getVariance operation.");
    }

    @Override
    public boolean isValid() throws IllegalAccessException {

        double sumProbability = 0;
        for (Integer integer : phis.keySet()) {
            double phi = phis.get(integer);
            if ((phi<0) || (phi>1)) {
                return false;
            }
            sumProbability += phi;
        }

        sumProbability += this.missingPhi;

        if ((sumProbability - 1) < THRESHOLD) {
            return true;
        }

        return false;
    }
}
