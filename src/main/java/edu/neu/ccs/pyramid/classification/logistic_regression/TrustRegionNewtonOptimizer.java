package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * the implementation is based on the liblinear package and the following paper
 * Lin, Chih-Jen, Ruby C. Weng, and S. Sathiya Keerthi.
 * "Trust region newton method for logistic regression."
 * The Journal of Machine Learning Research 9 (2008): 627-650.
 * Created by chengli on 11/27/14.
 */
public class TrustRegionNewtonOptimizer {
    private final RidgeLogisticLoss loss;

    private final double   eps;

    private final int maxIter;

    // Parameters for updating the iterates.
    private static final double ETA0 = 1e-4;
    private static final double ETA1 = 0.25;
    private static final double ETA2 = 0.75;
    // Parameters for updating the trust region size delta.
    private static final double SIGMA1 = 0.25;
    private static final double SIGMA2 = 0.5;
    private static final double SIGMA3 = 4;


    public TrustRegionNewtonOptimizer(final RidgeLogisticLoss loss) {
        this(loss, 0.1);
    }

    public TrustRegionNewtonOptimizer(final RidgeLogisticLoss loss, double eps) {
        this(loss, eps, 1000);
    }

    public TrustRegionNewtonOptimizer(final RidgeLogisticLoss loss, double eps, int maxIter) {
        this.loss = loss;
        this.eps = eps;
        this.maxIter = maxIter;
    }

    void tron(Vector w) {

        int numColumns = loss.getNumColumns();
        double delta, snorm, one = 1.0;
        double alpha, f, fnew, prered, actred, gs;
        int search = 1, iter = 1;

        Vector w_new = new DenseVector(numColumns);
        Vector g = new DenseVector(numColumns);

        for (int i = 0; i < numColumns; i++)
            w.set(i,0);

        f = loss.fun(w);
        loss.grad(w, g);
        delta = g.norm(2);
        double gnorm1 = delta;
        double gnorm = gnorm1;

        if (gnorm <= eps * gnorm1) search = 0;

        iter = 1;

        while (iter <= maxIter && search != 0) {

            Pair<Vector,Vector> result = trcg(delta, g);
            Vector s = result.getFirst();
            Vector r = result.getSecond();

            for (int j=0;j<w.size();j++){
                w_new.set(j,w.get(j));
            }
            daxpy(one, s, w_new);

            gs = g.dot(s);
            prered = -0.5 * (gs - s.dot(r));
            fnew = loss.fun(w_new);

            // Compute the actual reduction.
            actred = f - fnew;

            // On the first iteration, adjust the initial step bound.
            snorm = s.norm(2);
            if (iter == 1) delta = Math.min(delta, snorm);

            // Compute prediction alpha*snorm of the step.
            if (fnew - f - gs <= 0)
                alpha = SIGMA3;
            else
                alpha = Math.max(SIGMA1, -0.5 * (gs / (fnew - f - gs)));

            // Update the trust region bound according to the ratio of actual to
            // predicted reduction.
            if (actred < ETA0 * prered)
                delta = Math.min(Math.max(alpha, SIGMA1) * snorm, SIGMA2 * delta);
            else if (actred < ETA1 * prered)
                delta = Math.max(SIGMA1 * delta, Math.min(alpha * snorm, SIGMA2 * delta));
            else if (actred < ETA2 * prered)
                delta = Math.max(SIGMA1 * delta, Math.min(alpha * snorm, SIGMA3 * delta));
            else
                delta = Math.max(delta, Math.min(alpha * snorm, SIGMA3 * delta));

            System.out.println("f = "+f);

            if (actred > ETA0 * prered) {
                iter++;
                for (int j=0;j<w.size();j++){
                    w.set(j,w_new.get(j));
                }
                f = fnew;
                loss.grad(w, g);

                gnorm = g.norm(2);
                if (gnorm <= eps * gnorm1) break;
            }
            if (f < -1.0e+32) {

                break;
            }
            if (Math.abs(actred) <= 0 && prered <= 0) {
                System.out.println("WARNING: actred and prered <= 0%n");
                break;
            }
            if (Math.abs(actred) <= 1.0e-12 * Math.abs(f) && Math.abs(prered) <= 1.0e-12 * Math.abs(f)) {
                System.out.println("WARNING: actred and prered too small%n");
                break;
            }
        }
    }

    /**
     *
     * @param delta input
     * @param g input
     * @return s, r
     */
    private Pair<Vector,Vector> trcg(double delta, Vector g) {
        int numColumns = loss.getNumColumns();
        double one = 1;
        Vector d = new DenseVector(numColumns);
        Vector Hd = new DenseVector(numColumns);
        double rTr, rnewTrnew, cgtol;
        Vector s = new DenseVector(numColumns);
        Vector r = new DenseVector(numColumns);
        Pair<Vector,Vector> result = new Pair<>();
        for (int i = 0; i < numColumns; i++) {
            s.set(i,0);
            r.set(i,-g.get(i));
            d.set(i,r.get(i));
        }
        cgtol = 0.1 * g.norm(2);

        rTr = r.dot(r);

        while (true) {
            if (r.norm(2) <= cgtol) {
                break;
            }
            loss.Hv(d, Hd);

            double alpha = rTr / d.dot(Hd);
            daxpy(alpha, d, s);
            if (s.norm(2) > delta) {
                alpha = -alpha;
                daxpy(alpha, d, s);

                double std = s.dot(d);
                double sts = s.dot(s);
                double dtd = d.dot(d);
                double dsq = delta * delta;
                double rad = Math.sqrt(std * std + dtd * (dsq - sts));
                if (std >= 0)
                    alpha = (dsq - sts) / (std + rad);
                else
                    alpha = (rad - std) / dtd;
                daxpy(alpha, d, s);
                alpha = -alpha;
                daxpy(alpha, Hd, r);
                break;
            }
            alpha = -alpha;
            daxpy(alpha, Hd, r);
            rnewTrnew = r.dot(r);
            double beta = rnewTrnew / rTr;
            scale(beta, d);
            daxpy(one, r, d);
            rTr = rnewTrnew;
        }

        result.setFirst(s);
        result.setSecond(r);
        return result;
    }

    /**
     * constant times a vector plus a vector
     * vector2 += constant * vector1
     */
    private static void daxpy(double constant, Vector vector1, Vector vector2) {
        if (constant == 0) return;

        assert vector1.size() == vector2.size();
        for (int i = 0; i < vector1.size(); i++) {
            vector2.set(i, vector2.get(i) + constant * vector1.get(i));
        }
    }



    /**
     * scales a vector by a constant
     */
    private static void scale(double constant, Vector vector) {
        if (constant == 1.0) return;
        for (int i = 0; i < vector.size(); i++) {
            vector.set(i, vector.get(i) * constant);
        }

    }
}
