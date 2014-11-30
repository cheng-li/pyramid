package edu.neu.ccs.pyramid.classification.logistic_regression;

import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSet;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * ridge logistic regression loss function
 * to be minimized
 * Created by chengli on 11/27/14.
 */
public class RidgeLogisticLoss {

    /**
     * regularization constant
     */
    private final Vector regularization;
    /**
     * Xw
     */
    private final Vector scores;
    /**
     * diagonal of matrix
     */
    private final Vector diagonals;
    /**
     * the labels should be 1/-1
     */
    private final DataSet dataSet;
    private int[] labels;
    private int numRows;
    /**
     * including the bias
     */
    private int numColumns;

    public int getNumColumns() {
        return numColumns;
    }

    /**
     *
     * @param clfDataSet
     * @param regularization constant vector
     */
    public RidgeLogisticLoss(ClfDataSet clfDataSet, Vector regularization) {
        this.dataSet = clfDataSet;
        numRows = dataSet.getNumDataPoints();
        numColumns = dataSet.getNumFeatures() + 1;
        scores = new DenseVector(numRows);
        diagonals = new DenseVector(numRows);
        this.regularization = regularization;
        this.labels = changeLabels(clfDataSet);
    }

    /**
     * dot product of a row vector (adding the constant bias feature ) and another vector
     */
    private double rowDot(int rowIndex, Vector vector){
        double product = 0;
        // add bias
        product += vector.get(0);
        Vector part = vector.viewPart(1,vector.size()-1);
        product += dataSet.getRow(rowIndex).dot(part);
        return product;
    }


    private void Xv(Vector v, Vector Xv) {
        if (Xv.isDense()){
            IntStream.range(0,numRows).parallel()
                    .forEach(i -> Xv.set(i, rowDot(i,v)));
        } else {
            for (int i = 0; i < numRows; i++) {
                Xv.set(i, rowDot(i,v));
            }
        }

    }

    /**
     * dot product of a column vector and another vector
     * @param columnIndex the bias feature has index 0
     * @param vector
     * @return
     */
    private double columnDot(int columnIndex, Vector vector){
        if (columnIndex==0){
            return vector.zSum();
        } else {
            return dataSet.getColumn(columnIndex-1).dot(vector);
        }
    }

    private void XTv(Vector v, Vector XTv) {
        if (XTv.isDense()){
            IntStream.range(0,numColumns).parallel()
                    .forEach(i -> XTv.set(i,columnDot(i,v)));
        } else {
            for (int i=0;i<numColumns;i++){
                XTv.set(i,columnDot(i,v));
            }
        }

    }


    public double fun(Vector w) {
        double f = 0;
        Xv(w, scores);
        f += w.dot(w);
        f /= 2.0;
        for (int i = 0; i < numRows; i++) {
            double yz = labels[i] * scores.get(i);
            if (yz >= 0)
                f += regularization.get(i) * Math.log(1 + Math.exp(-yz));
            else
                f += regularization.get(i) * (-yz + Math.log(1 + Math.exp(yz)));
        }

        return (f);
    }

    public void grad(Vector w, Vector g) {

        int[] y = labels;
        for (int i = 0; i < numRows; i++) {
            scores.set(i, 1 / (1 + Math.exp(-y[i] * scores.get(i))));
            diagonals.set(i, scores.get(i) * (1 - scores.get(i)));
            scores.set(i, regularization.get(i) * (scores.get(i) - 1) * y[i]);
            //it seems that scores are messed up at this point of time
        }
        XTv(scores, g);

        for (int i=0;i<g.size();i++){
            g.set(i,w.get(i)+g.get(i));
        }
    }

    public void Hv(Vector s, Vector Hs) {

        Vector wa = new DenseVector(numRows);

        Xv(s, wa);
        for (int i = 0; i < numRows; i++)
            wa.set(i, regularization.get(i) * diagonals.get(i) * wa.get(i));

        XTv(wa, Hs);
        for (int i = 0; i < numColumns; i++)
            Hs.set(i,s.get(i) + Hs.get(i));
        // delete[] wa;
    }



//    public static DataSet addConstantColumn(ClfDataSet clfDataSet){
//        if (clfDataSet.hasMissingValue()){
//            throw new RuntimeException("cannot handle missing values in logistic regression");
//        }
//
//        DataSet dataSet1 = DataSetBuilder.getBuilder()
//                .numDataPoints(clfDataSet.getNumDataPoints())
//                .numFeatures(clfDataSet.getNumFeatures() + 1)
//                .dense(clfDataSet.isDense())
//                .missingValue(false)
//                .build();
//        for (int i=0;i<dataSet1.getNumDataPoints();i++){
//            //add constant 1
//            dataSet1.setFeatureValue(i,0,1);
//            //only copy non-zero elements
//            Vector vector = clfDataSet.getRow(i);
//            for (Vector.Element element: vector.nonZeroes()){
//                int featureIndex = element.index();
//                double value = element.get();
//                dataSet1.setFeatureValue(i,featureIndex+1,value);
//            }
//        }
//        return dataSet1;
//    }

    /**
     * change labels to 1/-1
     * @param clfDataSet
     * @return
     */
    public static int[] changeLabels(ClfDataSet clfDataSet){
        if (clfDataSet.getNumClasses()!=2){
            throw new RuntimeException("clfDataSet.getNumClasses()!=2");
        }
        int[] labels = clfDataSet.getLabels();
        int[] changed = new int[labels.length];
        for (int i=0;i<labels.length;i++){
            if (labels[i]==0){
                changed[i] = -1;
            } else {
                changed[i] = 1;
            }
        }
        return changed;
    }
}
