package edu.neu.ccs.pyramid.optimization;

/**
 * Created by chengli on 6/9/15.
 */
public interface Optimizer {
    void optimize();
    double getFinalObjective();
    Terminator getTerminator();

    interface Iterative extends Optimizer{
        void iterate();
    }
}
