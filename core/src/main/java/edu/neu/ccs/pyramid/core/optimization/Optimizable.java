package edu.neu.ccs.pyramid.core.optimization;

import org.apache.mahout.math.Vector;

/**
 * Created by chengli on 12/7/14.
 * inspired by the design of mallet
 */
public interface Optimizable{
    Vector getParameters();

    /**
     * we recommend using "=" rather than vector.assign()
     * @param parameters
     */
    void setParameters(Vector parameters);

    public interface ByValue extends Optimizable{
        double getValue();
    }

    public interface ByGradient extends Optimizable {
        Vector getGradient();
    }

    public interface ByGradientValue extends Optimizable.ByValue, Optimizable.ByGradient{

    }

}
