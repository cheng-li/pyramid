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

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Rainicy on 10/23/15.
 */
public class BMMClassifier implements MultiLabelClassifier, Serializable {
    private static final long serialVersionUID = 1L;
    int numLabels;
    int numClusters;
    int numSample = 100;

    String predictMode;

    // parameters
    // format: [numClusters][numLabels]
    LogisticRegression[][] binaryLogitRegressions;

    LogisticRegression softMaxRegression;


    // for predictions from single cluster sampling
    Set<MultiLabel> samplesForCluster;

    BufferedWriter bw = null;

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
        this.softMaxRegression = new LogisticRegression(numClusters, numFeatures,true);
        this.samplesForCluster = null;
        this.predictMode = "mixtureMax";
    }

    public BMMClassifier() {
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


    /**
     * return the log[p(y_n | z_n=k, x_n; w_k)] by all k from 1 to K.
     * @param logProbsForX
     * @param Y
     * @return
     */
    public double[] clusterConditionalLogProbArr(double[][][] logProbsForX, Vector Y) {
        double[] probArr = new double[numClusters];

        for (int k=0; k<numClusters; k++) {
            probArr[k] = clusterConditionalLogProb(logProbsForX, Y, k);
        }

        return probArr;
    }

    /**
     * return one value for log [p(y_n | z_n=k, x_n; w_k)] by given k;
     * @param logProbsForX
     * @param Y
     * @param k
     * @return
     */
    private double clusterConditionalLogProb(double[][][] logProbsForX, Vector Y, int k) {
        LogisticRegression[] logisticRegressionsK = binaryLogitRegressions[k];

        double logProbResult = 0.0;
        for (int l=0; l<logisticRegressionsK.length; l++) {
            if (Y.get(l) == 1.0) {
                logProbResult += logProbsForX[k][l][1];
            } else {
                logProbResult += logProbsForX[k][l][0];
            }
        }
        return logProbResult;
    }

    //TODO: remove
    public MultiLabel[] predict(MultiLabelClfDataSet dataSet) {
        List<MultiLabel> results = new ArrayList<>(dataSet.getNumDataPoints());
        File file = new File("/home/wang.bin/exps/pyramid/experiments/exp211Log/slashdot.log");

        try {
            bw = new BufferedWriter(new FileWriter(file));
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int n=0; n<dataSet.getNumDataPoints(); n++) {
            try {
                bw.write("data point: " + n + "\t" + "y: " + dataSet.getMultiLabels()[n].toString() + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }

            results.add(n, predict(dataSet.getRow(n)));
        }
        try {
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return results.toArray(new MultiLabel[results.size()]);

    }

    public MultiLabel predict(Vector vector) {
        MultiLabel predLabel = new MultiLabel();
        double maxLogProb = Double.NEGATIVE_INFINITY;
        Vector predVector = new DenseVector(numLabels);

        int[] clusters = IntStream.range(0, numClusters).toArray();
        double[] logisticLogProb = softMaxRegression.predictClassLogProbs(vector);
        double[] logisticProb = softMaxRegression.predictClassProbs(vector);
        EnumeratedIntegerDistribution enumeratedIntegerDistribution = new EnumeratedIntegerDistribution(clusters,logisticProb);

        //TODO: remove
//        File file = new File("/home/wang.bin/exps/pyramid/experiments/exp211Log/slashdot.log");
//        BufferedWriter bw = null;
//        try {
//            bw = new BufferedWriter(new FileWriter(file, true));
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        try {
            bw.write("PIs: \t");
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (double piK : logisticProb) {
            try {
                bw.write(piK + "\t");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            bw.write("\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
//        try {
//            bw.close();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        // cache the prediction for binaryLogitRegressions[numClusters][numLabels]
        double[][][] logProbsForX = new double[numClusters][numLabels][2];
        for (int k=0; k<logProbsForX.length; k++) {
            for (int l=0; l<logProbsForX[k].length; l++) {
                logProbsForX[k][l] = binaryLogitRegressions[k][l].predictClassLogProbs(vector);
            }
        }

        // samples methods
        if (predictMode.equals("mixtureMax")) {
            for (int s=0; s<numSample; s++) {
                int cluster = enumeratedIntegerDistribution.sample();
                Vector candidateY = new DenseVector(numLabels);

                for (int l=0; l<numLabels; l++) {
                    LogisticRegression regression = binaryLogitRegressions[cluster][l];
                    double prob = regression.predictClassProb(vector, 1);
                    BernoulliDistribution bernoulliDistribution = new BernoulliDistribution(prob);
                    candidateY.set(l, bernoulliDistribution.sample());
                }

                double logProb = logProbYnGivenXnLogisticProb(logisticLogProb, candidateY, logProbsForX);

                if (logProb >= maxLogProb) {
                    predVector = candidateY;
                    maxLogProb = logProb;
                }
            }
        } else if (predictMode.equals("singleTop")) {
            try {
                this.samplesForCluster = sampleFromSingles(vector, logisticProb);
            } catch (IOException e) {
                e.printStackTrace();
            }
            for (MultiLabel label : this.samplesForCluster) {
                Vector candidateY = new DenseVector(numLabels);
                for(int labelIndex : label.getMatchedLabels()) {
                    candidateY.set(labelIndex, 1.0);
                }

                double logProb = logProbYnGivenXnLogisticProb(logisticLogProb, candidateY, logProbsForX);

                if (logProb >= maxLogProb) {
                    predVector = candidateY;
                    maxLogProb = logProb;
                }
            }
        } else {
            throw new RuntimeException("Unknown predictMode: " + predictMode);
        }

        for (int l=0; l<numLabels; l++) {
            if (predVector.get(l) == 1.0) {
                predLabel.addLabel(l);
            }
        }
        return predLabel;
    }

    private Set<MultiLabel> sampleFromSingles(Vector vector, double[] logisticProb) throws IOException {
        int top = 20;
        Set<MultiLabel> samples = new HashSet<>();
        for (int k=0; k<binaryLogitRegressions.length; k++) {
            Set<MultiLabel> sample = sampleFromSingle(vector, top, k, logisticProb[k]);
            for (MultiLabel multiLabel : sample) {
                if (!samples.contains(multiLabel)) {
                    samples.add(multiLabel);
                }
            }
        }
        return samples;
    }

    private Set<MultiLabel> sampleFromSingle(Vector vector, int top, int k, double probK) throws IOException {
        Set<MultiLabel> sample = new HashSet<>();
        double maxProb = 1.0;

        MultiLabel label = new MultiLabel();
        Map<Integer, Double> labelAbsProbMap = new HashMap<>();
        Map<Integer, Double> labelProbMap = new HashMap<>();
        for (int l=0; l<binaryLogitRegressions[k].length; l++) {
            LogisticRegression logisticRegression = binaryLogitRegressions[k][l];
            double prob = logisticRegression.predictClassProbs(vector)[1];
            if (prob > 0.5) {
                label.addLabel(l);
                maxProb *= prob;
            } else {
                maxProb *= (1-prob);
            }
            double absProb = Math.abs(prob - 0.5);
            labelAbsProbMap.put(l, absProb);
            labelProbMap.put(l, prob);
        }
        MultiLabel copyLabel1 = new MultiLabel();
        for (int l : label.getMatchedLabels()) {
            copyLabel1.addLabel(l);
        }
        sample.add(copyLabel1);

        double prevProb = maxProb;
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
            double targetProb = labelProbMap.get(minL);
            // flip the label
            if (label.matchClass(minL)) {
                label.removeLabel(minL);
                prevProb = prevProb / targetProb * (1-targetProb);
            } else {
                label.addLabel(minL);
                prevProb = prevProb * targetProb / (1-targetProb);
            }
            labelAbsProbMap.remove(minL);

            // check if we need to stop sampling.
            if (prevProb < maxProb + 1 - 1.0/probK) {
                break;
            }

            MultiLabel copyLabel = new MultiLabel();
            for (int l : label.getMatchedLabels()) {
                copyLabel.addLabel(l);
            }
            sample.add(copyLabel);
        }

        return sample;
    }

    private double logProbYnGivenXnLogisticProb(double[] logisticLogProb, Vector Y, double[][][] logProbsForX) {
        double[] logPYnk = clusterConditionalLogProbArr(logProbsForX,Y);
        double[] sumLog = new double[logisticLogProb.length];
        for (int k=0; k<numClusters; k++) {
            sumLog[k] = logisticLogProb[k] + logPYnk[k];
        }


        // TODO: remove
        MultiLabel multiLabel = new MultiLabel();
        for (Vector.Element nonzero : Y.nonZeroes()) {
            multiLabel.addLabel(nonzero.index());
        }
        try {
            bw.write("predict: " + multiLabel.toString() + "\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
        //TODO: remove
        for (int k=0; k<numClusters; k++) {
            try {
                bw.write(Math.exp(logPYnk[k]) + "\t");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            bw.write(Math.exp(MathUtil.logSumExp(sumLog)) + "\n");
        } catch (IOException e) {
            e.printStackTrace();
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

    public void setPredictMode(String mode) {
        this.predictMode = mode;
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
}
