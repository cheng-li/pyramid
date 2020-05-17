package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.BMDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.multilabel_classification.plugin_rule.GeneralF1Predictor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.util.ArgMax;
import edu.neu.ccs.pyramid.util.ArgSort;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Vectors;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Reranker implements VectorCalibrator {
    private static final long serialVersionUID = 2L;
    Regressor regressor;
    int numCandidate;
    private PredictionFeatureExtractor predictionFeatureExtractor;
    private int minPredictionSize = 0;
    private int maxPredictionSize = Integer.MAX_VALUE;



    public Reranker(Regressor regressor, int numCandidate,
                    PredictionFeatureExtractor predictionFeatureExtractor) {
        this.regressor = regressor;
        this.numCandidate = numCandidate;
        this.predictionFeatureExtractor = predictionFeatureExtractor;
    }

    public Regressor getRegressor() {
        return regressor;
    }


    public void setMinPredictionSize(int minPredictionSize) {
        this.minPredictionSize = minPredictionSize;
    }

    public void setMaxPredictionSize(int maxPredictionSize) {
        this.maxPredictionSize = maxPredictionSize;
    }

    public double prob(Vector vector, MultiLabel multiLabel, double[] labelProbs){
        DynamicProgramming dynamicProgramming = new DynamicProgramming(labelProbs);
        List<Pair<MultiLabel,Double>> topK = dynamicProgramming.topK(numCandidate);

        PredictionCandidate predictionCandidate = new PredictionCandidate();
        predictionCandidate.x = vector;
        predictionCandidate.labelProbs = labelProbs;
        predictionCandidate.multiLabel = multiLabel;
        predictionCandidate.sparseJoint = topK;
        Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
        double score = regressor.predict(feature);
        if (score>1){
            score=1;
        }

        if (score<0){
            score=0;
        }
        return score;
    }


    public MultiLabel[] predict(DataSet dataSet, LabelProbMatrix labelProbMatrix){
        MultiLabel[] multiLabels = new MultiLabel[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel().forEach(i->{
            multiLabels[i]=predict(dataSet.getRow(i), Vectors.toArray(labelProbMatrix.getMatrix().getRow(i)));
        });
        return multiLabels;
    }

    public MultiLabel predict(Vector vector, double[] labelProbs) {

        DynamicProgramming dynamicProgramming = new DynamicProgramming(labelProbs);
        List<Pair<MultiLabel,Double>> sparseJoint = dynamicProgramming.topK(numCandidate);

        List<MultiLabel> multiLabels = sparseJoint.stream().map(pair->pair.getFirst())
                .filter(candidate->candidate.getNumMatchedLabels() >= minPredictionSize && candidate.getNumMatchedLabels() <= maxPredictionSize)
                .collect(Collectors.toList());
        List<Pair<MultiLabel,Double>> candidates = new ArrayList<>();

        if (multiLabels.isEmpty()){
            int[] sorted = ArgSort.argSortDescending(labelProbs);
            MultiLabel multiLabel = new MultiLabel();
            for (int i=0;i<minPredictionSize;i++){
                multiLabel.addLabel(sorted[i]);
            }
            multiLabels.add(multiLabel);
        }

        for (MultiLabel candidate: multiLabels){
            PredictionCandidate predictionCandidate = new PredictionCandidate();
            predictionCandidate.x = vector;
            predictionCandidate.labelProbs = labelProbs;
            predictionCandidate.multiLabel = candidate;
            predictionCandidate.sparseJoint = sparseJoint;
            Vector feature = predictionFeatureExtractor.extractFeatures(predictionCandidate);
            double score = regressor.predict(feature);
            candidates.add(new Pair<>(candidate,score));
        }

        Comparator<Pair<MultiLabel,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        return candidates.stream().max(comparator).map(Pair::getFirst).get();
    }

    @Override
    public double calibrate(Vector vector) {
        double score = regressor.predict(vector);
        if (score>1){
            score=1;
        }

        if (score<0){
            score=0;
        }
        return score;
    }


    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("Reranker{");
        sb.append("regressor=").append(regressor);
        sb.append('}');
        return sb.toString();
    }
}
