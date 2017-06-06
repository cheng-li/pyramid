package edu.neu.ccs.pyramid.core.optimization;

/**
 * Created by chengli on 6/9/15.
 */
public interface Optimizer {
    void optimize();
    double getFinalObjective();
    Terminator getTerminator();
}
