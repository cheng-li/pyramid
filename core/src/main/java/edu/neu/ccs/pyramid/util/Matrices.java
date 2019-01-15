package edu.neu.ccs.pyramid.util;

import org.ojalgo.matrix.PrimitiveMatrix;
import org.ojalgo.matrix.decomposition.Cholesky;
import org.ojalgo.matrix.store.MatrixStore;

/**
 * Created by chengli on 1/16/16.
 */
public class Matrices {
//    /**
//     * calculating determinant directly results in overflow or underflow
//     * use the method described in http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
//     * Cholesky decomposition
//     * @param matrix
//     * @return
//     */
//    public static double logDeterminant(BasicMatrix matrix){
//        Cholesky cholesky = Cholesky.makePrimitive();
//        cholesky.compute(matrix,true);
//        MatrixStore matrixStore = cholesky.getL();
//        double sum = 0;
//        for (int d=0;d<matrix.countColumns();d++){
//            double v = matrixStore.doubleValue(d,d);
//            sum += Math.log(v);
//        }
//        return 2*sum;
//    }


    public static String display(PrimitiveMatrix matrix) {
        long row = matrix.countRows();
        long column = matrix.countColumns();
        StringBuilder sb = new StringBuilder();
        sb.append("size = ").append(matrix.countRows()).append("x")
                .append(matrix.countColumns()).append("\n");
        for (int i=0;i<row;i++){
            for (int j=0;j<column;j++){
                sb.append(matrix.get(i,j));
                if (j!=column-1){
                    sb.append(", ");
                }
            }
            sb.append("\n");
        }
        return sb.toString();
    }
}
