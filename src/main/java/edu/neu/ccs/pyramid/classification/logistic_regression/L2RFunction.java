package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 11/27/14.
 */
public class L2RFunction {

    /**
     * regularization constant
     */
    private final Vector C;
    /**
     * Xw
     */
    private final Vector z;
    /**
     * diagonal of matrix
     */
    private final Vector D;
    /**
     * the first column should be all 1
     * the labels should be 1/-1
     */
    private final ClfDataSet dataSet;

    /**
     *
     * @param dataSet
     * @param C constant vector
     */
    public L2RFunction( ClfDataSet dataSet, Vector C ) {
        int l = dataSet.getNumDataPoints();
        this.dataSet = dataSet;
        z = new DenseVector(l);
        D = new DenseVector(l);
        this.C = C;
    }


    private void Xv(Vector v, Vector Xv) {
        for (int i = 0; i < dataSet.getNumDataPoints(); i++) {
            Xv.set(i, dataSet.getRow(i).dot(v));
        }
    }

    private void XTv(Vector v, Vector XTv) {
        for (int i=0;i<dataSet.getNumFeatures();i++){
            XTv.set(i,dataSet.getColumn(i).dot(v));
        }
    }


    public double fun(Vector w) {
        double f = 0;
        int[] y = dataSet.getLabels();
        int l = dataSet.getNumDataPoints();
        Xv(w, z);
        f += w.dot(w);
        f /= 2.0;
        for (int i = 0; i < l; i++) {
            double yz = y[i] * z.get(i);
            if (yz >= 0)
                f += C.get(i) * Math.log(1 + Math.exp(-yz));
            else
                f += C.get(i) * (-yz + Math.log(1 + Math.exp(yz)));
        }

        return (f);
    }

    public void grad(Vector w, Vector g) {

        int[] y = dataSet.getLabels();
        int l = dataSet.getNumDataPoints();
        for (int i = 0; i < l; i++) {
            z.set(i, 1 / (1 + Math.exp(-y[i] * z.get(i))));
            D.set(i,z.get(i) * (1 - z.get(i)));
            z.set(i,C.get(i) * (z.get(i) - 1) * y[i]);
            //it seems that z is messed up at this point of time
        }
        XTv(z, g);

        for (int i=0;i<g.size();i++){
            g.set(i,w.get(i)+g.get(i));
        }
    }

    public void Hv(Vector s, Vector Hs) {

        int l = dataSet.getNumDataPoints();
        int w_size = get_nr_variable();
        Vector wa = new DenseVector(l);

        Xv(s, wa);
        for (int i = 0; i < l; i++)
            wa.set(i, C.get(i) * D.get(i) * wa.get(i));

        XTv(wa, Hs);
        for (int i = 0; i < w_size; i++)
            Hs.set(i,s.get(i) + Hs.get(i));
        // delete[] wa;
    }

    public int get_nr_variable() {
        return dataSet.getNumFeatures();
    }
}
