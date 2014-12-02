package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import org.apache.commons.math3.util.FastMath;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.IOException;

/**
 * Created by Rainicy on 12/2/14.
 */
public class KDE {
    public static void main(String[] args) throws IOException, ClassNotFoundException {
        double sigma = Double.parseDouble(args[0]);
        String kernel = args[1];
        System.out.println(sigma);
        System.out.println(kernel);
        testKDE(sigma, kernel);
    }

    private static void testKDE(double sigma, String kernel) throws IOException, ClassNotFoundException {

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File("/home/bingyu/spame/zscore/train.trec"),
                DataSetType.CLF_DENSE, true);
        ClfDataSet testDataset = TRECFormat.loadClfDataSet(new File("/home/bingyu/spame/zscore/test.trec"),
                DataSetType.CLF_DENSE, true);

        int numClasses = dataSet.getNumClasses();
        int numDataSet = dataSet.getNumDataPoints();
        int[] counts = new int[numClasses];
        double[] prior = new double[numClasses];
        System.out.println("Calculating prior...");
        // set prior
        for (int i=0; i<numDataSet; i++) {
            counts[dataSet.getLabels()[i]]++;
            prior[dataSet.getLabels()[i]] += 1.0;
        }
        for (int j=0; j<numClasses; j++) {
            prior[j] /= (double)numDataSet;
        }

        int numTestDataSet = testDataset.getNumDataPoints();
        int[] predicts = new int[numTestDataSet];
        System.out.println("Predicting...");
        // predict
        for (int i=0; i<numTestDataSet; i++) {
            System.out.print(i + ", ");
            Vector z = testDataset.getRow(i);
            double[] K = new double[numClasses];
            for (int j=0; j<numDataSet; j++) {
                int label = dataSet.getLabels()[j];
                Vector x = dataSet.getRow(j);
                double kValue = getKernelValue(z, x, kernel, sigma);
                K[label] += kValue;
            }
            for (int j=0; j<numClasses; j++) {
                K[j] /= (double) counts[j];
                K[j] *= prior[j];
            }
            int predict = 0;
            double maxProb = K[0];
            for(int j=1; j<numClasses; j++) {
                if(maxProb < K[j]) {
                    maxProb = K[j];
                    predict = j;
                }
            }
            predicts[i] = predict;
        }

        System.out.println("\nAccuracy: " + Accuracy.accuracy(testDataset.getLabels(), predicts));

    }
    private static double getKernelValue(Vector vectorI, Vector vectorJ, String kernelName, double sigma) {
//        System.out.println("++" + kernel + "--" + sigma);
        double result;
        if (kernelName.equalsIgnoreCase("linear")) {
            result = vectorI.dot(vectorJ);
        }
        else if (kernelName.equalsIgnoreCase("rbf")) {
            // default sigma
            Vector diffVector = vectorI.minus(vectorJ);
            double diffValue = diffVector.dot(diffVector);
            diffValue = (-0.5) * diffValue / FastMath.pow(sigma, 2);
            result = FastMath.exp(diffValue);
        }
        else {
            throw new RuntimeException("The kernel cannot be recognized.");
        }
        return result;
    }
}
