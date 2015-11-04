package edu.neu.ccs.pyramid.optimization;

/**
 * Created by chengli on 11/3/15.
 */
public interface Parallelizable {
    void setParallelism(boolean isParallel);
    boolean isParallel();
}
