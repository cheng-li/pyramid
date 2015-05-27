package edu.neu.ccs.pyramid.classification.naive_bayes;

/**
 * Created by Rainicy on 10/10/14.
 *
 * Multinomial Distribution.
 *
 * @see edu.neu.ccs.pyramid.classification.naive_bayes.Distribution
 *
 * @reference
 * http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
 *
 */
public class Multinomial implements Distribution {

    private double theta;
    private double logPosTheta;
    private double logNegTheta;


    /**
     * Constructor by given Nyi: the number of times feature i appears
     * in a sample of class y in the training set; Ny: the total count
     * of all features for class y; and n: the number of features.
     *
     * @param Nyi
     * @param Ny
     * @param n
     *
     * http://scikit-learn.org/stable/modules/naive_bayes.html
     */
    public Multinomial(int Nyi, int Ny, int n) {
        this.theta = (double) (Nyi + 1) / (Ny + n);
        this.logPosTheta = Math.log(theta);
        this.logNegTheta = Math.log(1 - theta);
    }

    @Override
    public double probability(double x) throws IllegalArgumentException {
        return Math.exp(logProbability(x));
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        if (x > 0.0) {
            return logPosTheta;
        }
        return logNegTheta;
    }
}
