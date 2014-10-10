package edu.neu.ccs.pyramid.classification.naive_bayes;

import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.util.FastMath;

import java.util.Arrays;

/**
 * Created by Rainicy on 10/9/14.
 *
 * Gamma distribution implements Distribution.
 *
 * @see edu.neu.ccs.pyramid.classification.naive_bayes.Distribution
 *
 * @refrence
 *  1) org.apache.commons.math3.distribution.GammaDistribution
 *  2) http://research.microsoft.com/en-us/um/people/minka/papers/minka-gamma.pdf
 */
public class Gamma implements Distribution {

    /** Shape, also called alpha. */
    private double shape;
    /** Scale, also called beta. */
    private double scale;



    /** Gamma constructor by given shape and scale. */
    public Gamma(double shape, double scale) {
        if (shape <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SHAPE, shape);
        }
        if (scale <=0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SCALE, scale);
        }

    }

    /** Gamma constructor by given variables */
    public Gamma(double[] variables) {
        fit(variables);
    }

    /**
     * Using the approximated shape(alpha) and scale(beta).
     * @refrence
     *  http://research.microsoft.com/en-us/um/people/minka/papers/minka-gamma.pdf
     * @param variables
     * @throws IllegalArgumentException
     */
    @Override
    public void fit(double[] variables) throws IllegalArgumentException {
        double xMean = StatUtils.mean(variables);
        double standardDeviation = FastMath.sqrt(StatUtils.variance(variables));
        double meanDivideStandardDev = xMean/standardDeviation;
        shape = meanDivideStandardDev * meanDivideStandardDev;
        scale = standardDeviation * standardDeviation / xMean;

        // http://www.ncl.ucar.edu/Document/Functions/Built-in/dim_gamfit_n.shtml
//        double meanSqar = StatUtils.sumSq(variables)/variables.length;
//        double xMean = StatUtils.mean(variables);
//        shape = xMean * xMean / (meanSqar - xMean * xMean);
//        scale = (meanSqar - xMean * xMean) / xMean;

        System.out.println("Shape: " + shape + "; Scale: " + scale);
        if (shape <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SHAPE, shape);
        }
        if (scale <=0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SCALE, scale);
        }
    }

    @Override
    public double probability(double x) throws IllegalArgumentException {
        if (x < 0) {
            return 0;
        }
//        return FastMath.exp(logProbability(x));

        double x1 = FastMath.pow(x, shape-1.0);
        double x2 = - (x / scale);
        double x3 = FastMath.exp(x2);
        double x4 = FastMath.pow(scale, shape);
        double x5 = gamma(x);
        System.out.println(x5);
        return x1 * x3 / x4 / x5;
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        if (x < 0) {
            return Double.NEGATIVE_INFINITY;
        }
//        double scalePowShape = FastMath.pow(scale, shape);
//        double xSumScalePowShape = FastMath.pow((x+scale), shape);
//        System.out.println(scalePowShape);

//        return scalePowShape / xSumScalePowShape;
//        double logScale = FastMath.log(scale);
//        double logXSumScale = FastMath.log(x + scale);
//        double shapeTimes = shape * logScale - shape * logXSumScale;
////        System.out.println(FastMath.exp(shapeTimes));
//        return shapeTimes;


        return FastMath.log(probability(x));
    }

    @Override
    // TODO
    public double cumulativeProbability(double x) {
        return 0;
    }

    @Override
    public double getMean() throws IllegalAccessException {
        return shape * scale;
    }

    @Override
    public double getVariance() throws IllegalAccessException {
        return shape * scale * scale;
    }

    @Override
    public boolean isValid() throws IllegalAccessException {
        throw new IllegalAccessException("Cannot support this function in" +
                "Gamma distribution.");
    }

    private double gamma(double x) {
        double natureX = FastMath.exp(-x);
        double xPower = FastMath.pow(x, x-0.5);
        double sqrtPi = FastMath.sqrt(2.0 * FastMath.PI);
        return sqrtPi * natureX * xPower;
    }

    public double getShape() {
        return shape;
    }

    public double getAlpha() {
        return shape;
    }

    public double getScale() {
        return scale;
    }

    public double getBeta() {
        return scale;
    }


}
