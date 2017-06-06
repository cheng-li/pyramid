package edu.neu.ccs.pyramid.core.classification;

import edu.neu.ccs.pyramid.core.feature.FeatureList;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.DataSet;
import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.core.util.ArgMax;
import edu.neu.ccs.pyramid.core.util.MathUtil;
import org.apache.mahout.math.Vector;

import java.util.Arrays;

/**
 * Created by chengli on 8/19/14.
 */
public class PriorProbClassifier implements Classifier.ProbabilityEstimator {
    private static final long serialVersionUID = 2L;

    private int numClasses;
    private double[] counts;
    private double[] probs;
    private int topClass;
    private FeatureList featureList;
    private LabelTranslator labelTranslator;


    public PriorProbClassifier(int numClasses) {
        this.numClasses = numClasses;
        this.probs = new double[numClasses];
        this.counts = new double[numClasses];
    }

    public void fit(ClfDataSet clfDataSet){
        int[] labels = clfDataSet.getLabels();
        for (int label: labels){
            counts[label] += 1;
        }
        int numDataPoints = clfDataSet.getNumDataPoints();
        for (int k=0;k<this.numClasses;k++){
            this.probs[k] = counts[k]/numDataPoints;
        }

        this.topClass = ArgMax.argMax(probs);
    }

    public void fit(DataSet dataSet, double[][] targetDistribution, double[] weights){
        double totalCount = MathUtil.arraySum(weights);

        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int k=0;k<numClasses;k++){
                counts[k] += targetDistribution[i][k]*weights[i];
            }
        }
        for (int k=0;k<this.numClasses;k++){
            this.probs[k] = counts[k]/totalCount;
        }
        this.topClass = ArgMax.argMax(probs);
    }

    /**
     * gradient for class k
     * @param k
     * @return
     */
    public double[] getGradient(ClfDataSet clfDataSet, int k){
        int numDataPoints = clfDataSet.getNumDataPoints();
        double[] gradient = new double[numDataPoints];
        for (int i=0;i<numDataPoints;i++){
            int label = clfDataSet.getLabels()[i];
            if (label==k){
                gradient[i] = 1- probs[label];
            } else {
                gradient[i] = 0 - probs[label];
            }
        }
        return gradient;
    }

    @Override
    public int predict(Vector vector) {
        return this.topClass;
    }

    @Override
    public int getNumClasses() {
        return this.numClasses;
    }

    @Override
    public double[] predictClassProbs(Vector vector) {
        return this.probs;
    }

    public double[] getClassProbs(){
        return this.probs;
    }

    @Override
    public String toString() {
        return "PriorProbClassifier{" +
                "numClasses=" + numClasses +
                ", probs=" + Arrays.toString(probs) +
                ", topClass=" + topClass +
                '}';
    }

    @Override
    public FeatureList getFeatureList() {
        return featureList;
    }

    public void setFeatureList(FeatureList featureList) {
        this.featureList = featureList;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return labelTranslator;
    }

    public void setLabelTranslator(LabelTranslator labelTranslator) {
        this.labelTranslator = labelTranslator;
    }

}
