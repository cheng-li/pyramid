package edu.neu.ccs.pyramid.classification.naive_bayes;

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

    public static final double LANCZOS_G = 607.0 / 128.0;
    private static final double[] LANCZOS = {
            0.99999999999999709182,
            57.156235665862923517,
            -59.597960355475491248,
            14.136097974741747174,
            -0.49191381609762019978,
            .33994649984811888699e-4,
            .46523628927048575665e-4,
            -.98374475304879564677e-4,
            .15808870322491248884e-3,
            -.21026444172410488319e-3,
            .21743961811521264320e-3,
            -.16431810653676389022e-3,
            .84418223983852743293e-4,
            -.26190838401581408670e-4,
            .36899182659531622704e-5,
    };

    /** Shape, also called alpha. */
    private double shape;
    /** Scale, also called beta. */
    private double scale;

    private double minY;
    private double maxLogY;
    private double shiftedShape;
    private double densityPrefactor2;
    private double densityPrefactor1;
    private double logDensityPrefactor1;
    private double logDensityPrefactor2;

    public static double lanczos(final double x) {
        double sum = 0.0;
        for (int i = LANCZOS.length - 1; i > 0; --i) {
            sum += LANCZOS[i] / (x + i);
        }
        return sum + LANCZOS[0];
    }

    /** Gamma constructor by given shape and scale. */
    public Gamma(double shape, double scale) {
        if (shape <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SHAPE, shape);
        }
        if (scale <=0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SCALE, scale);
        }
        this.shape = shape;
        this.scale = scale;
        this.shiftedShape = shape + Gamma.LANCZOS_G + 0.5;
        final double aux = FastMath.E / (2.0 * FastMath.PI * shiftedShape);
        this.densityPrefactor2 = shape * FastMath.sqrt(aux) / Gamma.lanczos(shape);
        this.logDensityPrefactor2 = FastMath.log(shape) + 0.5 * FastMath.log(aux) -
                FastMath.log(Gamma.lanczos(shape));
        this.densityPrefactor1 = this.densityPrefactor2 / scale *
                FastMath.pow(shiftedShape, -shape) *
                FastMath.exp(shape + Gamma.LANCZOS_G);
        this.logDensityPrefactor1 = this.logDensityPrefactor2 - FastMath.log(scale) -
                FastMath.log(shiftedShape) * shape +
                shape + Gamma.LANCZOS_G;
        this.minY = shape + Gamma.LANCZOS_G - FastMath.log(Double.MAX_VALUE);
        this.maxLogY = FastMath.log(Double.MAX_VALUE) / (shape - 1.0);

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
        double[] logX = new double[variables.length];
        for (int i=0; i<variables.length; i++) {
            logX[i] = FastMath.log(variables[i]);
        }
        double logXMean = StatUtils.mean(logX);

        shape = 0.5 / (FastMath.log(xMean) - logXMean);
        scale = xMean / shape;

        if (shape <= 0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SHAPE, shape);
        }
        if (scale <=0) {
            throw new NotStrictlyPositiveException(LocalizedFormats.SCALE, scale);
        }

        this.shiftedShape = shape + Gamma.LANCZOS_G + 0.5;
        final double aux = FastMath.E / (2.0 * FastMath.PI * shiftedShape);
        this.densityPrefactor2 = shape * FastMath.sqrt(aux) / Gamma.lanczos(shape);
        this.logDensityPrefactor2 = FastMath.log(shape) + 0.5 * FastMath.log(aux) -
                FastMath.log(Gamma.lanczos(shape));
        this.densityPrefactor1 = this.densityPrefactor2 / scale *
                FastMath.pow(shiftedShape, -shape) *
                FastMath.exp(shape + Gamma.LANCZOS_G);
        this.logDensityPrefactor1 = this.logDensityPrefactor2 - FastMath.log(scale) -
                FastMath.log(shiftedShape) * shape +
                shape + Gamma.LANCZOS_G;
        this.minY = shape + Gamma.LANCZOS_G - FastMath.log(Double.MAX_VALUE);
        this.maxLogY = FastMath.log(Double.MAX_VALUE) / (shape - 1.0);

    }

    @Override
    public double probability(double x) throws IllegalArgumentException {
        if (x < 0) {
            return 0;
        }
        final double y = x / scale;
        if ((y <= minY) || (FastMath.log(y) >= maxLogY)) {
            /*
             * Overflow.
             */
            final double aux1 = (y - shiftedShape) / shiftedShape;
            final double aux2 = shape * (FastMath.log1p(aux1) - aux1);
            final double aux3 = -y * (Gamma.LANCZOS_G + 0.5) / shiftedShape +
                    Gamma.LANCZOS_G + aux2;
            return densityPrefactor2 / x * FastMath.exp(aux3);
        }
        /*
         * Natural calculation.
         */
        return densityPrefactor1 * FastMath.exp(-y) * FastMath.pow(y, shape - 1);
    }

    @Override
    public double logProbability(double x) throws IllegalArgumentException {
        if (x < 0) {
            return Double.NEGATIVE_INFINITY;
        }
        final double y = x / scale;
        if ((y <= minY) || (FastMath.log(y) >= maxLogY)) {
            /*
             * Overflow.
             */
            final double aux1 = (y - shiftedShape) / shiftedShape;
            final double aux2 = shape * (FastMath.log1p(aux1) - aux1);
            final double aux3 = -y * (Gamma.LANCZOS_G + 0.5) / shiftedShape +
                    Gamma.LANCZOS_G + aux2;
            return logDensityPrefactor2 - FastMath.log(x) + aux3;
        }
        /*
         * Natural calculation.
         */
        return logDensityPrefactor1 - y + FastMath.log(y) * (shape - 1);
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
