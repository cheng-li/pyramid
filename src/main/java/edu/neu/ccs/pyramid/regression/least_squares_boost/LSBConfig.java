package edu.neu.ccs.pyramid.regression.least_squares_boost;

import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.util.Sampling;

import java.util.stream.IntStream;

/**
 * Created by chengli on 6/3/15.
 */
public class LSBConfig {

    private double learningRate;
    private RegressorFactory regressorFactory;

    public double getLearningRate() {
        return learningRate;
    }

    public RegressorFactory getRegressorFactory() {
        return regressorFactory;
    }

    public static Builder getBuilder(){
        return new Builder();
    }


    public static class Builder {
        double learningRate = 1;
        RegressorFactory regressorFactory = new RegTreeFactory(new RegTreeConfig());

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder setRegressorFactory(RegressorFactory regressorFactory) {
            this.regressorFactory = regressorFactory;
            return this;
        }

        public LSBConfig build() {
            return new LSBConfig(this);
        }
    }



    //PRIVATE
    private LSBConfig(Builder builder) {
        this.learningRate = builder.learningRate;
        this.regressorFactory = builder.regressorFactory;
    }
}
