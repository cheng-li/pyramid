package edu.neu.ccs.pyramid.core.classification.l2boost;

import edu.neu.ccs.pyramid.core.classification.Classifier;
import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.util.MathUtil;
import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.core.optimization.gradient_boosting.GradientBoosting;
import org.apache.mahout.math.Vector;

/**
 * binary logistic gradient boosting
 * Created by chengli on 10/1/15.
 */
public class L2Boost extends GradientBoosting implements Classifier.ScoreEstimator, Classifier.ProbabilityEstimator{

    private FeatureList featureList;
    private LabelTranslator labelTranslator;

    /**
     * 1 ensemble; used for scoring positive class
     */
    public L2Boost() {
        super(1);
    }

    @Override
    public double[] predictClassProbs(Vector vector) {
        double[] scores = this.predictClassScores(vector);
        return predictClassProbs(scores);
    }

    double[] predictClassProbs(double[] scores){
        double[] probVector = new double[2];
        double logDenominator = MathUtil.logSumExp(scores);
        for (int k=0;k<2;k++){

            double logNumerator = scores[k];
            double pro = Math.exp(logNumerator-logDenominator);
            probVector[k]=pro;
        }
        return probVector;
    }

    @Override
    public double predictClassScore(Vector vector, int k) {
        if (k==0){
            return 0;
        } else {
            return getEnsemble(0).score(vector);
        }
    }

    @Override
    public int getNumClasses() {
        return 2;
    }

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }
}
