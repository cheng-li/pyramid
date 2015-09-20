package edu.neu.ccs.pyramid.multilabel_classification.sampling;

import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBConfig;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBTrainer;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.Accuracy;
import org.apache.mahout.math.Vector;

import java.io.IOException;


/**
 * Created by Rainicy on 9/7/15.
 */
public class GibbsSamplingTrainer {
    private GibbsSamplingConfig config;
    private GibbsSampling sampling;
    private int numClasses;

    /**
     * Constructor.
     * @param config
     * @param sampling
     */
    public GibbsSamplingTrainer(GibbsSamplingConfig config, GibbsSampling sampling) {
        if(config.getDataSet().getNumClasses() != sampling.getNumClasses()) {
            throw new IllegalArgumentException("config.getDataSet().getNumClasses() != sampling.getNumClasses()");
        }

        this.config = config;
        this.sampling = sampling;
        MultiLabelClfDataSet dataSet = config.getDataSet();
        sampling.setFeatureList(dataSet.getFeatureList());
        sampling.setLabelTranslator(dataSet.getLabelTranslator());
        this.numClasses = dataSet.getNumClasses();
    }

    /**
     * Train the GibbsSampling, using LKTreeBoost as the basic classifier.
     */
    public void train() {

        System.out.println("starting Gibbs Sampling Training ...");
        // train a classifier for each label.
        for (int l=0; l<numClasses; l++) {
            // first initialize a new ClfDataSet.
            ClfDataSet dataSet = initDataSetforK(config.getDataSet(), l);

            // initialize the LKTBTrainConfig
            LKTBConfig trainConfig = generateConfig(dataSet);

            // add a new classifier
            LKTreeBoost lkTreeBoost = new LKTreeBoost(2);

            LKTBTrainer trainer = new LKTBTrainer(trainConfig, lkTreeBoost);
            for (int round=0; round<config.getNumRounds(); round++) {
                System.out.println("label = " + (l+1) + "/" + numClasses +
                        "; round = " + (round+1) + "/" + config.getNumRounds());
                trainer.iterate();
            }

            this.sampling.addClassifier(lkTreeBoost, l);

            System.out.println("label = " + (l+1) +"; accuracy = " + Accuracy.accuracy(lkTreeBoost,dataSet));
        }
    }

    /**
     * By given the ClfDataSet, generate an LKTBConfig.
     * @param dataSet
     * @return
     */
    private LKTBConfig generateConfig(ClfDataSet dataSet) {
        LKTBConfig trainConfig = new LKTBConfig.Builder(dataSet)
                .numLeaves(config.getNumLeaves())
                .learningRate(config.getLearningRate())
                .numSplitIntervals(config.getNumSplitIntervals())
                .minDataPerLeaf(config.getMinDataPerLeaf())
                .dataSamplingRate(1)
                .featureSamplingRate(1)
                .randomLevel(config.getRandomLevel())
                .considerHardTree(config.isConsiderHardTree())
                .considerExpectationTree(config.isConsiderExpectationTree())
                .considerProbabilisticTree(config.isConsiderProbabilisticTree())
                .setLeafOutputType(config.getLeafOutputType())
                .build();
        return trainConfig;
    }

    /**
     * The goal of this function is to transform a MultiLabelClfDataSet into
     * a binary ClfDataSet.
     * By given the original MultiLabelClfDataSet and given label index,
     * put all labels except for the given label into feature matrix. Then label
     * this new raw data as 1 if the given label appears in the MultiLabel, otherwise
     * 0.
     * @param dataSet
     * @param label
     * @return
     */
    private ClfDataSet initDataSetforK(MultiLabelClfDataSet dataSet, int label) {
        int numDataPoints = dataSet.getNumDataPoints();
        int origNumFeatures = dataSet.getNumFeatures();
        int numFeatures = origNumFeatures + numClasses - 1;

        ClfDataSet clfDataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(numDataPoints).numFeatures(numFeatures)
                .dense(dataSet.isDense()).missingValue(dataSet.hasMissingValue())
                .numClasses(2).build();


        // put multilabel into feature matrix and update the label
        MultiLabel[] multiLabels = dataSet.getMultiLabels();
        for (int i=0; i<numDataPoints; i++) {
            // set original features first.
            Vector vector = dataSet.getRow(i);
            for (Vector.Element element : vector.nonZeroes()) {
                clfDataSet.setFeatureValue(i, element.index(), element.get());
            }

            // set the new features from labels second.
            MultiLabel multiLabel = multiLabels[i];
            for (Integer j : multiLabel.getMatchedLabels()) {
                if (j < label) { // label is on the left hand of given removing label.
                    clfDataSet.setFeatureValue(i, (origNumFeatures+j), 1.0);
                } else if (j > label) { // on the right hand
                    clfDataSet.setFeatureValue(i, (origNumFeatures+j-1), 1.0);
                }
            }
            // assign the new label to this datapoint.
            clfDataSet.setLabel(i, multiLabel.matchClass(label) ? 1 : 0);
        }
        return clfDataSet;
    }

}