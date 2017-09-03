package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.eval.InstanceAverage;
import edu.neu.ccs.pyramid.eval.MLConfusionMatrix;
import edu.neu.ccs.pyramid.util.Sampling;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by chengli on 4/7/16.
 */
public class LossMatrixGenerator {
    public static Matrix matrix(int n, String lossName){
        int size = (int)Math.pow(2,n);
        double[][] matrixBuilder = new double[size][size];
        for (int i=0;i<size;i++) {
            for (int j = 0; j < size; j++) {
                String ib = toBinary(i,n);
                String jb = toBinary(j,n);
                MultiLabel multiLabel1 = toML(ib);
                MultiLabel multiLabel2 = toML(jb);
                MultiLabel[] trueLabels = {multiLabel1};
                MultiLabel[] predicted = {multiLabel2};
                MLConfusionMatrix mlConfusionMatrix = new MLConfusionMatrix(n,trueLabels,predicted);
                InstanceAverage instanceAverage = new InstanceAverage(mlConfusionMatrix);
                double loss;
                switch (lossName.toLowerCase()){
                    case "hamming":
                        loss = instanceAverage.getHammingLoss()*n;
                        break;
                    case "overlap":
                        loss = 1-instanceAverage.getOverlap();
                        break;
                    case "accuracy":
                        loss = 1-instanceAverage.getAccuracy();
                        break;
                    case "precision":
                        loss = 1-instanceAverage.getPrecision();
                        break;
                    case "recall":
                        loss = 1-instanceAverage.getRecall();
                        break;
                    case "f1":
                        loss = 1-instanceAverage.getF1();
                        break;
                    default:
                        throw new IllegalArgumentException("unknown loss");
                }

                matrixBuilder[i][j]=loss;
            }
        }
        Matrix matrix = new DenseMatrix(matrixBuilder);
        return matrix;
    }

    private static String toBinary(int number, int length){
        String iBinary = Integer.toBinaryString(number);
        StringBuilder sb = new StringBuilder();
        for (int l=0;l<length-iBinary.length();l++){
            sb.append("0");
        }
        sb.append(iBinary);
        String ib = sb.toString();
        return ib;
    }

    private static MultiLabel toML(String str){
        MultiLabel multiLabel = new MultiLabel();
        for (int i=0;i<str.length();i++){
            String sub = str.substring(i,i+1);
            if (sub.equals("1")){
                multiLabel.addLabel(i);
            }
        }
        return multiLabel;
    }

    public static List<Double> sampleDistribution(int numLabels){
        int size = (int)Math.pow(2,numLabels);
        List<Double> list = new ArrayList<>();
        double used = 0;
        for (int i=0;i<size;i++){

            double prob;
            if (i==size-1){
                prob = 1-used;
            } else {
                prob = Sampling.doubleUniform(0,1-used);;
            }

            list.add(prob);
            used += prob;
        }
        return list;
    }
}
