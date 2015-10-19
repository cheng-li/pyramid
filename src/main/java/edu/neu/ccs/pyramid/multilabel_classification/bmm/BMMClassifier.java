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


import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/7/15.
 */
public class BMMClassifier implements MultiLabelClassifier, Serializable {
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
        UniformRealDistribution uniform = new UniformRealDistribution(0,1.0);
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
        double[] logisticLogProb = logisticRegression.predictClassLogProbs(vector);
        double[] logisticProb = logisticRegression.predictClassProbs(vector);
        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters,logisticProb);
        for (int s=0; s<numSample; s++) {
            int cluster = enumeratedIntegerDistribution.sample();
            Vector candidateVector = new DenseVector(numLabels);

            for (int l=0; l<numLabels; l++) {
                candidateVector.set(l, distributions[cluster][l].sample());
            }

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


    public Set<MultiLabel> sampleFromSingles() throws IOException {
        int top = 10;
        Set<MultiLabel> samples = new HashSet<>();
        for (int k=0; k<distributions.length; k++) {
            Set<MultiLabel> sample = sampleFromSingle(distributions[k], top);
            for (MultiLabel multiLabel : sample) {
                if (!samples.contains(multiLabel)) {
                    samples.add(multiLabel);
                }
            }
        }
        return samples;
    }

    private Set<MultiLabel> sampleFromSingle(BernoulliDistribution[] distribution, int top) throws IOException {

        Set<MultiLabel> sample = new HashSet<>();

        MultiLabel label = new MultiLabel();
        Map<Integer, Double> labelAbsProbMap = new HashMap<>();
        for (int l=0; l<distribution.length; l++) {
            BernoulliDistribution bd = distribution[l];
            double prob = bd.getP();
            if (prob > 0.5) {
                label.addLabel(l);
            }
            double absProb = Math.abs(prob - 0.5);
            labelAbsProbMap.put(l, absProb);
        }
        MultiLabel copyLabel1 = new MultiLabel();
        for (int l : label.getMatchedLabels()) {
            copyLabel1.addLabel(l);
        }
        sample.add(copyLabel1);

        for (int i=1; i<top; i++) {
            // find min abs prob among all labels
            int minL = 0;
            double minProb = 100.0;
            for (Map.Entry<Integer, Double> entry : labelAbsProbMap.entrySet()) {
                if (entry.getValue() < minProb) {
                    minL = entry.getKey();
                    minProb = entry.getValue();
                }
            }
            // flip the label
            if (label.matchClass(minL)) {
                label.removeLabel(minL);
            } else {
                label.addLabel(minL);
            }
            labelAbsProbMap.remove(minL);

            MultiLabel copyLabel = new MultiLabel();
            for (int l : label.getMatchedLabels()) {
                copyLabel.addLabel(l);
            }
            sample.add(copyLabel);
        }

        return sample;
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

    @Override
    public void serialize(File file) throws Exception {
        File parent = file.getParentFile();
        if (!parent.exists()) {
            parent.mkdir();
        }
        try (
                FileOutputStream fileOutputStream = new FileOutputStream(file);
                BufferedOutputStream bufferedOutputStream = new BufferedOutputStream(fileOutputStream);
                ObjectOutputStream objectOutputStream = new ObjectOutputStream(bufferedOutputStream);
                ){
            objectOutputStream.writeObject(this);
        }
    }

    @Override
    public void serialize(String file) throws Exception {
        File file1 = new File(file);
        serialize(file1);
    }

    public static BMMClassifier deserialize(File file) throws Exception {
        try (
                FileInputStream fileInputStream = new FileInputStream(file);
                BufferedInputStream bufferedInputStream = new BufferedInputStream(fileInputStream);
                ObjectInputStream objectInputStream = new ObjectInputStream(bufferedInputStream);
                ){
            BMMClassifier bmmClassifier = (BMMClassifier) objectInputStream.readObject();
            return bmmClassifier;
        }
    }

    public static BMMClassifier deserialize(String file) throws Exception {
        File file1 = new File(file);
        return deserialize(file1);
    }
}
