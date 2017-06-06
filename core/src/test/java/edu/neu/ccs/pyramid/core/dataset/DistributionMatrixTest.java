package edu.neu.ccs.pyramid.core.dataset;

public class DistributionMatrixTest {
    public static void main(String[] args) {
        WeightMatrix matrix = new WeightMatrix(2,2);
        matrix.setProbability(0,0,1);
        matrix.setProbability(0,1,2);
        matrix.setProbability(1,0,3);
        matrix.setProbability(1,1,4);
        matrix.normalize();
        System.out.println(matrix);
    }

}