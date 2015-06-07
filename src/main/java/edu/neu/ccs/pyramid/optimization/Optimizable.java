package edu.neu.ccs.pyramid.optimization;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 12/7/14.
 * inspired by the design of mallet
 */
public interface Optimizable {
    Vector getParameters();
    void setParameters(Vector parameters);
    Optimizable newInstance(Vector parameters);

    public interface ByGradient extends Optimizable {
        Vector getGradient();
        Optimizable.ByGradient newInstance(Vector parameters);
    }

    public interface ByGradientValue extends Optimizable{
        Vector getGradient();
        double getValue();
        Optimizable.ByGradientValue newInstance(Vector parameters);

    }

}
