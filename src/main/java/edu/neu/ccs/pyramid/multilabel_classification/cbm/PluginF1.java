package edu.neu.ccs.pyramid.multilabel_classification.cbm;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.multilabel_classification.PluginPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by chengli on 4/9/16.
 */
public class PluginF1 implements PluginPredictor<CBM>{
    CBM cbm;
    private String predictionMode = "support";
    private int numSamples = 1000;
    private List<MultiLabel> support;
    private double probMassThreshold = 0.95;

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

    public void setSupport(List<MultiLabel> support) {
        this.support = support;
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
            case "samplingNonEmpty":
                pred =  predictBySamplingNonEmpty(vector);
                break;
            default:
                throw new IllegalArgumentException("unknown mode");
        }
        return pred;
    }

    private MultiLabel predictBySampling(Vector vector){
//        List<MultiLabel> samples = cbm.samples(vector, numSamples);
        Pair<List<MultiLabel>, List<Double>> pair = cbm.samples(vector, probMassThreshold);
        return GeneralF1Predictor.predict(cbm.getNumClasses(),pair.getFirst(), pair.getSecond());
    }


    private MultiLabel predictBySamplingNonEmpty(Vector vector){
        Pair<List<MultiLabel>, List<Double>> pair = cbm.sampleNonEmptySets(vector, probMassThreshold);
        return GeneralF1Predictor.predict(cbm.getNumClasses(),pair.getFirst(), pair.getSecond());
    }

    private MultiLabel predictBySupport(Vector vector){
        List<Double> probs = cbm.predictAssignmentProbs(vector,support);
        return GeneralF1Predictor.predict(cbm.getNumClasses(),support,probs);
    }


    public MultiLabel showPredictBySampling(Vector vector){
        System.out.println("sampling procedure");
//        List<MultiLabel> samples = cbm.samples(vector, numSamples);
        Pair<List<MultiLabel>, List<Double>> pair = cbm.samples(vector, probMassThreshold);
        List<Pair<MultiLabel, Double>> list = new ArrayList<>();
        List<MultiLabel> labels = pair.getFirst();
        List<Double> probs = pair.getSecond();
        for (int i=0;i<labels.size();i++){
            list.add(new Pair<>(labels.get(i),probs.get(i)));
        }
        Comparator<Pair<MultiLabel, Double>> comparator = Comparator.comparing(a-> a.getSecond());

        System.out.println(list.stream().sorted(comparator.reversed()).collect(Collectors.toList()));



//        for (int i=0;i<labels.size();i++){
//            System.out.println(labels.get(i)+": "+probs.get(i));
//        }
        return GeneralF1Predictor.predict(cbm.getNumClasses(),pair.getFirst(), pair.getSecond());
    }

    public void showPredictBySamplingNonEmpty(Vector vector){
        System.out.println("sampling procedure");
        Pair<List<MultiLabel>, List<Double>> pair = cbm.sampleNonEmptySets(vector, probMassThreshold);
        List<Pair<MultiLabel, Double>> list = new ArrayList<>();
        List<MultiLabel> labels = pair.getFirst();
        List<Double> probs = pair.getSecond();
        double[] probsArray = probs.stream().mapToDouble(a->a).toArray();

        for (int i=0;i<labels.size();i++){
            list.add(new Pair<>(labels.get(i),probs.get(i)));
        }
        Comparator<Pair<MultiLabel, Double>> comparator = Comparator.comparing(a-> a.getSecond());

        MultiLabel gfmPred =  GeneralF1Predictor.predict(cbm.getNumClasses(),pair.getFirst(), pair.getSecond());
        MultiLabel argmaxPre = cbm.predict(vector);
        System.out.println("expected f1 of argmax predictor= "+GeneralF1Predictor.expectedF1(labels,probsArray, argmaxPre,cbm.getNumClasses()));
        System.out.println("expected f1 of GFM predictor= "+GeneralF1Predictor.expectedF1(labels,probsArray, gfmPred,cbm.getNumClasses()));

        System.out.println(list.stream().sorted(comparator.reversed()).collect(Collectors.toList()));
    }

    public GeneralF1Predictor.Analysis showPredictBySupport(Vector vector, MultiLabel truth){
//        System.out.println("support procedure");
        List<Double> probs = cbm.predictAssignmentProbs(vector,support);
        double[] probArray = probs.stream().mapToDouble(a->a).toArray();

        MultiLabel prediction =  GeneralF1Predictor.predict(cbm.getNumClasses(),support,probs);
        GeneralF1Predictor.Analysis analysis = GeneralF1Predictor.showSupportPrediction(support,probArray, truth, prediction, cbm.getNumClasses());
        return analysis;
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
