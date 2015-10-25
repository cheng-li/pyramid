package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;

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

import java.io.Serializable;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 10/23/15.
 */
public class BMMClassifier implements MultiLabelClassifier, Serializable {
    private static final long serialVersionUID = 1L;
    int numLabels;
    int numClusters;
    int numSample = 100;

    // parameters
    // format: [numClusters][numLabels]
    LogisticRegression[][] binaryLogitRegressions;

    LogisticRegression softMaxRegression;

    /**
     * Default constructor by given a MultiLabelClfDataSet
     * @param dataSet
     * @param numClusters
     */
    public BMMClassifier(MultiLabelClfDataSet dataSet, int numClusters) {
        this(dataSet.getNumClasses(), numClusters, dataSet.getNumFeatures());
    }

    public BMMClassifier(int numClasses, int numClusters, int numFeatures) {
        this.numLabels = numClasses;
        this.numClusters = numClusters;
        // initialize distributions
        this.binaryLogitRegressions = new LogisticRegression[numClusters][numClasses];
        for (int k=0; k<numClusters; k++) {
            for (int l=0; l<numClasses; l++) {
                this.binaryLogitRegressions[k][l] = new LogisticRegression(2,numFeatures);
            }
        }
        this.softMaxRegression = new LogisticRegression(numClusters, numFeatures);
    }

    @Override
    public int getNumClasses() {
        return this.numLabels;
    }


    /**
     * return the log[p(y_n | z_n=k, x_n; w_k)] by all k from 1 to K.
     * @param X
     * @param Y
     * @return
     */
    public double[] clusterConditionalLogProbArr(Vector X, Vector Y) {
        double[] probArr = new double[numClusters];

        for (int k=0; k<numClusters; k++) {
            probArr[k] = clusterConditionalLogProb(X, Y, k);
        }

        return probArr;
    }

    /**
     * return one value for log [p(y_n | z_n=k, x_n; w_k)] by given k;
     * @param X
     * @param Y
     * @param k
     * @return
     */
    private double clusterConditionalLogProb(Vector X, Vector Y, int k) {
        LogisticRegression[] logisticRegressionsK = binaryLogitRegressions[k];

        double logProbResult = 0.0;
        for (int l=0; l<logisticRegressionsK.length; l++) {
            double[] logProbs = logisticRegressionsK[l].predictClassLogProbs(X);
            if (Y.get(l) == 1.0) {
                logProbResult += logProbs[1];
            } else {
                logProbResult += logProbs[0];
            }
        }
        return logProbResult;
    }

    @Override
    public MultiLabel predict(Vector vector) {
        double maxLogProb = Double.NEGATIVE_INFINITY;
        Vector predVector = new DenseVector(numLabels);

        int[] clusters = IntStream.range(0, numClusters).toArray();
        double[] logisticLogProb = softMaxRegression.predictClassLogProbs(vector);
        double[] logisticProb = softMaxRegression.predictClassProbs(vector);
        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters,logisticProb);
        for (int s=0; s<numSample; s++) {
            int cluster = enumeratedIntegerDistribution.sample();
            Vector candidateY = new DenseVector(numLabels);

            for (int l=0; l<numLabels; l++) {
                LogisticRegression regression = binaryLogitRegressions[cluster][l];
                double prob = regression.predictClassProb(vector, 1);
                BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(prob);
                candidateY.set(l, bernoulliDistribution.sample());
            }

            double logProb = logProbYnGivenXnLogisticProb(logisticLogProb, candidateY, vector);

            if (logProb >= maxLogProb) {
                predVector = candidateY;
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

    private double logProbYnGivenXnLogisticProb(double[] logisticLogProb, Vector Y, Vector X) {
        double[] logPYnk = clusterConditionalLogProbArr(X,Y);
        double[] sumLog = new double[logisticLogProb.length];
        for (int k=0; k<numClusters; k++) {
            sumLog[k] = logisticLogProb[k] + logPYnk[k];
        }
        return MathUtil.logSumExp(sumLog);
    }


    public String toString() {
        Vector vector = new RandomAccessSparseVector(softMaxRegression.getNumFeatures());
        double[] mixtureCoefficients = softMaxRegression.predictClassProbs(vector);
        final StringBuilder sb = new StringBuilder("BMM{\n");
        sb.append("numLabels=").append(numLabels).append("\n");
        sb.append("numClusters=").append(numClusters).append("\n");
        for (int k=0;k<numClusters;k++){
            sb.append("cluster ").append(k).append(":\n");
            sb.append("proportion = ").append(mixtureCoefficients[k]).append("\n");
        }
        sb.append('}');
        return sb.toString();
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    @Override
    public LabelTranslator getLabelTranslator() {
        return null;
    }

    public void setNumSample(int numSample) {
        this.numSample = numSample;
    }
}
