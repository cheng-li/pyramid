package edu.neu.ccs.pyramid.classification.logistic_regression;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 11/27/14.
 */
public class Tron {
    private final L2RFunction fun_obj;

    private final double   eps;

    private final int      max_iter;

    public Tron( final L2RFunction fun_obj ) {
        this(fun_obj, 0.1);
    }

    public Tron( final L2RFunction fun_obj, double eps ) {
        this(fun_obj, eps, 1000);
    }

    public Tron( final L2RFunction fun_obj, double eps, int max_iter ) {
        this.fun_obj = fun_obj;
        this.eps = eps;
        this.max_iter = max_iter;
    }

    void tron(Vector w) {
        // Parameters for updating the iterates.
        double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

        // Parameters for updating the trust region size delta.
        double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

        int n = fun_obj.get_nr_variable();
        int  cg_iter;
        double delta, snorm, one = 1.0;
        double alpha, f, fnew, prered, actred, gs;
        int search = 1, iter = 1;
        Vector s = new DenseVector(n);
        Vector r = new DenseVector(n);
        Vector w_new = new DenseVector(n);
        Vector g = new DenseVector(n);

        for (int i = 0; i < n; i++)
            w.set(i,0);

        f = fun_obj.fun(w);
        fun_obj.grad(w, g);
        delta = g.norm(2);
        double gnorm1 = delta;
        double gnorm = gnorm1;

        if (gnorm <= eps * gnorm1) search = 0;

        iter = 1;

        while (iter <= max_iter && search != 0) {
            cg_iter = trcg(delta, g, s, r);
            for (int j=0;j<w.size();j++){
                w_new.set(j,w.get(j));
            }
            daxpy(one, s, w_new);

            gs = g.dot(s);
            prered = -0.5 * (gs - s.dot(r));
            fnew = fun_obj.fun(w_new);

            // Compute the actual reduction.
            actred = f - fnew;

            // On the first iteration, adjust the initial step bound.
            snorm = s.norm(2);
            if (iter == 1) delta = Math.min(delta, snorm);

            // Compute prediction alpha*snorm of the step.
            if (fnew - f - gs <= 0)
                alpha = sigma3;
            else
                alpha = Math.max(sigma1, -0.5 * (gs / (fnew - f - gs)));

            // Update the trust region bound according to the ratio of actual to
            // predicted reduction.
            if (actred < eta0 * prered)
                delta = Math.min(Math.max(alpha, sigma1) * snorm, sigma2 * delta);
            else if (actred < eta1 * prered)
                delta = Math.max(sigma1 * delta, Math.min(alpha * snorm, sigma2 * delta));
            else if (actred < eta2 * prered)
                delta = Math.max(sigma1 * delta, Math.min(alpha * snorm, sigma3 * delta));
            else
                delta = Math.max(delta, Math.min(alpha * snorm, sigma3 * delta));

            System.out.println("f = "+f);

            if (actred > eta0 * prered) {
                iter++;
                for (int j=0;j<w.size();j++){
                    w.set(j,w_new.get(j));
                }
                f = fnew;
                fun_obj.grad(w, g);

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

    private int trcg(double delta, Vector g, Vector s, Vector r) {
        int n = fun_obj.get_nr_variable();
        double one = 1;
        Vector d = new DenseVector(n);
        Vector Hd = new DenseVector(n);
        double rTr, rnewTrnew, cgtol;

        for (int i = 0; i < n; i++) {
            s.set(i,0);
            r.set(i,-g.get(i));
            d.set(i,r.get(i));
        }
        cgtol = 0.1 * g.norm(2);

        int cg_iter = 0;
        rTr = r.dot(r);

        while (true) {
            if (r.norm(2) <= cgtol) break;
            cg_iter++;
            fun_obj.Hv(d, Hd);

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

        return (cg_iter);
    }

    /**
     * constant times a vector plus a vector
     *
     * <pre>
     * vector2 += constant * vector1
     * </pre>
     *
     * @since 1.8
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
     *
     * @since 1.8
     */
    private static void scale(double constant, Vector vector) {
        if (constant == 1.0) return;
        for (int i = 0; i < vector.size(); i++) {
            vector.set(i, vector.get(i) * constant);
        }

    }
}
