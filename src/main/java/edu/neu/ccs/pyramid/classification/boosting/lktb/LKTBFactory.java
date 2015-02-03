package edu.neu.ccs.pyramid.classification.boosting.lktb;

import edu.neu.ccs.pyramid.classification.Classifier;
import edu.neu.ccs.pyramid.classification.ClassifierFactory;
import edu.neu.ccs.pyramid.classification.TrainConfig;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;

import java.io.*;

/**
 * Created by chengli on 10/4/14.
 */
public class LKTBFactory implements ClassifierFactory{

    @Override
    public Classifier train(ClfDataSet dataSet, TrainConfig config) {
        if (!(config instanceof LKTBTrainConfig)){
            throw new IllegalArgumentException("!(config instanceof LKTBTrainConfig)");
        }
        LKTBTrainConfig lktbTrainConfig = (LKTBTrainConfig)config;
        return train(dataSet,lktbTrainConfig);
    }

    public LKTreeBoost train(ClfDataSet dataSet, LKTBTrainConfig config){
        LKTBConfig lktbConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(config.getNumLeaves())
                .learningRate(config.getLearningRate())
                .minDataPerLeaf(config.getMinDataPerLeaf())
                .dataSamplingRate(config.getDataSamplingRate())
                .featureSamplingRate(config.getFeatureSamplingRate())
                .numSplitIntervals(config.getNumSplitIntervals())
                .usePrior(config.usePrior())
                .build();
        LKTreeBoost lkTreeBoost = new LKTreeBoost(dataSet.getNumClasses());
        LKTBTrainer trainer  = new LKTBTrainer(lktbConfig,lkTreeBoost);
        for (int iteration=0;iteration<config.getNumIterations();iteration++){
            trainer.iterate();
        }
        return lkTreeBoost;
    }



    @Override
    public Classifier deserialize(File file) throws Exception {
        return LKTreeBoost.deserialize(file);
    }
}
