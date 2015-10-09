package edu.neu.ccs.pyramid.multilabel_classification.bmm;


import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.math3.distribution.BinomialDistribution;
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
    int numLabels;
    int numClusters;
    int numSample = 100;
    /**
     * format:[cluster][label]
     */
    BinomialDistribution[][] distributions;
    LogisticRegression logisticRegression;
    Mode mode = Mode.SAMPLING;


    public BMMClassifier(int numLabels, int numClusters, int numFeatures) {
        this.numLabels = numLabels;
        this.numClusters = numClusters;
        this.distributions = new BinomialDistribution[numClusters][numLabels];
        // random initialization
        UniformRealDistribution uniform = new UniformRealDistribution(0.25,0.75);
        for (int k=0;k<numClusters;k++){
            for (int l=0;l<numLabels;l++){
                double p = uniform.sample();
                distributions[k][l] = new BinomialDistribution(1,p);
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



    public double probYnGivenXnYn(Vector vectorX, Vector vectorY) {
        double[] logisticProb = logisticRegression.predictClassProbs(vectorX);
        return probYnGivenXnLogisticProb(logisticProb, vectorY);
    }

    public double probYnGivenXnLogisticProb(double[] logisticProb, Vector labelVector) {
        double prob = 0.0;
        double[] pYnk = clusterConditionalProbArr(labelVector);
        for (int k=0; k<numClusters; k++) {
            prob += logisticProb[k] * pYnk[k];
        }

        return prob;
    }

    @Override
    public MultiLabel predict(Vector vector) {

        double maxProb = Double.NEGATIVE_INFINITY;
        Vector predVector = new DenseVector(numLabels);

        for (int s=0; s<numSample; s++) {
            double[] logisticProb = logisticRegression.predictClassProbs(vector);
            int[] clusters = IntStream.range(0, numClusters).toArray();
            EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters,logisticProb);
            int cluster = enumeratedIntegerDistribution.sample();

            Vector candidateVector = new DenseVector(numLabels);

            for (int l=0; l<numLabels; l++) {
                candidateVector.set(l, distributions[cluster][l].sample());
            }

            double prob = probYnGivenXnLogisticProb(logisticProb, candidateVector);

            if (prob >= maxProb) {
                predVector = candidateVector;
                maxProb = prob;
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


    public double clusterConditionalProb(Vector vector, int clusterIndex){
        double prob = 1;
        for (int l=0;l< numLabels;l++){
            BinomialDistribution distribution = distributions[clusterIndex][l];
            prob *= distribution.probability((int)vector.get(l));
        }
        return prob;
    }

    /**
     * return the clusterConditionalProb for each cluster.
     * @param vector
     * @return
     */
    public double[] clusterConditionalProbArr(Vector vector){
        double[] probArr = new double[numClusters];

        for (int clusterIndex=0; clusterIndex<numClusters; clusterIndex++) {
            probArr[clusterIndex] = clusterConditionalProb(vector, clusterIndex);
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
                sb.append(d).append(":").append(distributions[k][d].getProbabilityOfSuccess());
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
