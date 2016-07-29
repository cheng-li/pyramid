package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 4/9/16.
 */
public class PluginF1 implements PluginPredictor<CBM>{
    CBM cbm;
    private String predictionMode = "sampling";
    private int numSamples = 1000;
    private List<MultiLabel> support;

    public PluginF1(CBM model) {
        this.cbm = model;
    }

    public PluginF1(CBM cbm, List<MultiLabel> support) {
        this.cbm = cbm;
        this.support = support;
    }

    public void setNumSamples(int numSamples) {
        this.numSamples = numSamples;
    }

    public void setPredictionMode(String predictionMode) {
        this.predictionMode = predictionMode;
    }

    public String getPredictionMode() {
        return predictionMode;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        MultiLabel pred = null;
        switch (predictionMode){
            case "support":
                pred =  predictBySupport(vector);
                break;
            case "sampling":
                pred =  predictBySampling(vector);
                break;
            default:
                throw new IllegalArgumentException("unknown mode");
        }
        return pred;
    }

    private MultiLabel predictBySampling(Vector vector){
        List<MultiLabel> samples = cbm.samples(vector, numSamples);
        return GeneralF1Predictor.predict(cbm.getNumClasses(),samples);
    }

    private MultiLabel predictBySupport(Vector vector){
        List<Double> probs = cbm.predictAssignmentProbs(vector,support);
        return GeneralF1Predictor.predict(cbm.getNumClasses(),support,probs);
    }


    @Override
    public CBM getModel() {
        return cbm;
    }

    public Matrix getPMatrix(Vector vector){
        List<MultiLabel> samples = cbm.samples(vector, numSamples);
        return GeneralF1Predictor.getPMatrix(cbm.getNumClasses(),samples);
    }
}
