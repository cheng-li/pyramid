package edu.neu.ccs.pyramid.optimization;

/**
 * Created by chengli on 6/9/15.
 */
public interface Optimizer {
    void optimize();
    void setCheckConvergence(boolean checkConvergence);
    void setMaxIteration(int maxIteration);
}
