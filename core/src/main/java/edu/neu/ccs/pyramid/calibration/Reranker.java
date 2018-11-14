package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.AccPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.BMDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.Vector;

import java.util.*;

public class Reranker implements MultiLabelClassifier, VectorCalibrator {
    private static final long serialVersionUID = 1L;
    Regressor regressor;
    CBM cbm;
    int numCandidate;
    private PredictionVectorizer predictionVectorizer;

    public Reranker(Regressor regressor, CBM cbm, int numCandidate, PredictionVectorizer predictionVectorizer) {
        this.regressor = regressor;
        this.cbm = cbm;
        this.numCandidate = numCandidate;
        this.predictionVectorizer = predictionVectorizer;
    }

    @Override
    public int getNumClasses() {
        return cbm.getNumClasses();
    }

    public double prob(Vector labelFeatures){
        double score = regressor.predict(labelFeatures);
        if (score>1){
            score=1;
        }

        if (score<0){
            score=0;
        }
        return score;
    }

    public double prob(Vector vector, MultiLabel multiLabel){
        double[] marginals = predictionVectorizer.getLabelCalibrator().calibratedClassProbs(cbm.predictClassProbs(vector));
        BMDistribution bmDistribution = cbm.computeBM(vector,0.001);
        Vector feature = predictionVectorizer.feature(bmDistribution, multiLabel,marginals,Optional.empty());
        double score = regressor.predict(feature);
        if (score>1){
            score=1;
        }

        if (score<0){
            score=0;
        }
        return score;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        double[] marginals = predictionVectorizer.getLabelCalibrator().calibratedClassProbs(cbm.predictClassProbs(vector));
        Map<MultiLabel,Integer> positionMap = PredictionVectorizer.positionMap(marginals);
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        List<Pair<MultiLabel,Double>> candidates = new ArrayList<>();
        BMDistribution bmDistribution = cbm.computeBM(vector,0.001);
        for (int i=0;i<numCandidate;i++){
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            Vector feature = predictionVectorizer.feature(bmDistribution, candidate,marginals,Optional.of(positionMap));
            double score = regressor.predict(feature);
            candidates.add(new Pair<>(candidate,score));
        }
        AccPredictor accPredictor = new AccPredictor(cbm);
        accPredictor.setComponentContributionThreshold(0.001);
        MultiLabel cbmPre = accPredictor.predict(vector);
        Vector feature = predictionVectorizer.feature(bmDistribution, cbmPre,marginals,Optional.of(positionMap));
        double score = regressor.predict(feature);
        candidates.add(new Pair<>(cbmPre,score));


        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return candidates.stream().max(comparator).map(pair->pair.getFirst()).get();
    }


    public boolean isInTopK(Vector vector,  MultiLabel groundTruth){
        double[] marginals = predictionVectorizer.getLabelCalibrator().calibratedClassProbs(cbm.predictClassProbs(vector));
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        for (int i=0;i<numCandidate;i++) {
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            if (candidate.equals(groundTruth)){
                return true;
            }
        }
        return false;
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return null;
    }

    @Override
    public double calibrate(Vector vector) {
        double score = regressor.predict(vector);
        return score;
    }
}
