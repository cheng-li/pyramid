package edu.neu.ccs.pyramid.multilabel_classification.bmm;


import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.BernoulliDistribution;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;


import java.util.stream.IntStream;

/**
 * Created by chengli on 10/7/15.
 */
public class BMMClassifier implements MultiLabelClassifier {
    private static final long serialVersionUID = 1L;
    int numLabels;
    int numClusters;
    int numSample = 100;
    /**
     * format:[cluster][label]
     */
    BernoulliDistribution[][] distributions;
    LogisticRegression logisticRegression;
    Mode mode = Mode.SAMPLING;


    public BMMClassifier(int numLabels, int numClusters, int numFeatures) {
        this.numLabels = numLabels;
        this.numClusters = numClusters;
        this.distributions = new BernoulliDistribution[numClusters][numLabels];
        // random initialization
        UniformRealDistribution uniform = new UniformRealDistribution(0.25,0.75);
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numLabels;l++){
                double p = uniform.sample();
                distributions[k][l] = new BernoulliDistribution(p);
            }
        }
        // num classes in logistic regression = num clusters
        this.logisticRegression = new LogisticRegression(numClusters, numFeatures);
    }

    public BMMClassifier(MultiLabelClfDataSet dataSet, int numClusters){
        this(dataSet.getNumClasses(),numClusters,dataSet.getNumFeatures());
    }

    public void initialize(){

    }

    @Override
    public int getNumClasses() {
        return this.numLabels;
    }


//
//    public double probYnGivenXnYn(Vector vectorX, Vector vectorY) {
//        double[] logisticProb = logisticRegression.predictClassProbs(vectorX);
//        return probYnGivenXnLogisticProb(logisticProb, vectorY);
//    }
//
//    public double probYnGivenXnLogisticProb(double[] logisticProb, Vector labelVector) {
//        double prob = 0.0;
//        double[] pYnk = clusterConditionalProbArr(labelVector);
//        for (int k=0; k<numClusters; k++) {
//            prob += logisticProb[k] * pYnk[k];
//        }
//
//        return prob;
//    }

    public double logProbYnGivenXnLogisticProb(double[] logisticLogProb, Vector labelVector) {
        double[] logPYnk = clusterConditionalLogProbArr(labelVector);
        double[] sumLog = new double[logisticLogProb.length];
        for (int k=0; k<numClusters; k++) {
            sumLog[k] = logisticLogProb[k] + logPYnk[k];
        }

        return MathUtil.logSumExp(sumLog);
    }

    @Override
    public MultiLabel predict(Vector vector) {

        double maxLogProb = Double.NEGATIVE_INFINITY;
        Vector predVector = new DenseVector(numLabels);

        int[] clusters = IntStream.range(0, numClusters).toArray();
        for (int s=0; s<numSample; s++) {
            double[] logisticProb = logisticRegression.predictClassProbs(vector);
            EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters,logisticProb);
            int cluster = enumeratedIntegerDistribution.sample();

            Vector candidateVector = new DenseVector(numLabels);

            for (int l=0; l<numLabels; l++) {
                candidateVector.set(l, distributions[cluster][l].sample());
            }

            double[] logisticLogProb = logisticRegression.predictClassLogProbs(vector);
            double logProb = logProbYnGivenXnLogisticProb(logisticLogProb, candidateVector);

            if (logProb >= maxLogProb) {
                predVector = candidateVector;
                maxLogProb = logProb;
            }
        }
        MultiLabel predLabel = new MultiLabel();
        for (int l=0; l<numLabels; l++) {
            if (predVector.get(l) == 1.0) {
                predLabel.addLabel(l);
            }
        }
        return predLabel;
    }

    public int getNumSample() {
        return numSample;
    }

    public void setNumSample(int numSample) {
        this.numSample = numSample;
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return null;
    }



    public double clusterConditionalLogProb(Vector vector, int clusterIndex){
        double logProb = 0.0;
        for (int l=0;l< numLabels;l++){
            BernoulliDistribution distribution = distributions[clusterIndex][l];
            logProb += distribution.logProbability((int)vector.get(l));
        }
        return logProb;
    }

    /**
     * return the clusterConditionalLogProb for each cluster.
     * @param vector
     * @return
     */
    public double[] clusterConditionalLogProbArr(Vector vector){
        double[] probArr = new double[numClusters];

        for (int clusterIndex=0; clusterIndex<numClusters; clusterIndex++) {
            probArr[clusterIndex] = clusterConditionalLogProb(vector, clusterIndex);
        }
        return probArr;
    }


    public String toString() {
        Vector vector = new RandomAccessSparseVector(logisticRegression.getNumFeatures());
        double[] mixtureCoefficients = logisticRegression.predictClassProbs(vector);
        final StringBuilder sb = new StringBuilder("BMM{\n");
        sb.append("numLabels=").append(numLabels).append("\n");
        sb.append("numClusters=").append(numClusters).append("\n");
        for (int k=0;k<numClusters;k++){
            sb.append("cluster ").append(k).append(":\n");
            sb.append("proportion = ").append(mixtureCoefficients[k]).append("\n");
            sb.append("probabilities = ").append("[");
            for (int d= 0; d < numLabels;d++){
                sb.append(d).append(":").append(distributions[k][d].getP());
                if (d!=numLabels-1){
                    sb.append(", ");
                }
            }
            sb.append("]\n");
        }
        sb.append('}');
        return sb.toString();
    }

    public enum Mode{
        SAMPLING, CRF
    }
}
