package edu.neu.ccs.pyramid.core.feature_selection;

import edu.neu.ccs.pyramid.core.util.EmpiricalCDF;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.List;

public class FusedKolmogorovFilterTest {
    public static void main(String[] args) {
        test2();

    }

    private static void test1(){
        Vector vector = new DenseVector(10);
        vector.set(0,0.1);
        vector.set(1,0.2);
        vector.set(2,0.15);
        vector.set(3,0.4);
        vector.set(4,0.7);
        vector.set(8,0.9);
        int[] labels = new int[10];
        labels[0] = 0 ;
        labels[1] = 1;
        labels[2] = 1;
        labels[3] = 1;
        labels[9] = 1;
        FusedKolmogorovFilter filter = new FusedKolmogorovFilter();
        filter.setNumBins(10);
        List<List<Double>> inputsEachClass = filter.generateInputsEachClass(vector, labels, 2);
        System.out.println(inputsEachClass);
        List<EmpiricalCDF> empiricalCDFs = filter.generateCDFs(vector,inputsEachClass);
        System.out.println(empiricalCDFs);
        System.out.println(filter.maxDistance(empiricalCDFs));
    }

    private static void test2(){
        Vector vector = new DenseVector(10);
        vector.set(0,0.1);
        vector.set(1,0.2);
        vector.set(2,0.15);
        vector.set(3,0.4);
        vector.set(4,0.7);
        vector.set(8,0.9);
        vector.set(9,0.8);
        int[] labels = new int[10];
        labels[0] = 0 ;
        labels[1] = 1;
        labels[2] = 2;
        labels[3] = 1;
        labels[9] = 2;
        FusedKolmogorovFilter filter = new FusedKolmogorovFilter();
        filter.setNumBins(10);
        List<List<Double>> inputsEachClass = filter.generateInputsEachClass(vector, labels, 3);
        System.out.println(inputsEachClass);
        List<EmpiricalCDF> empiricalCDFs = filter.generateCDFs(vector,inputsEachClass);
        System.out.println(empiricalCDFs);
        System.out.println(filter.maxDistance(empiricalCDFs));
        System.out.println(EmpiricalCDF.distance(empiricalCDFs.get(0),empiricalCDFs.get(1)));
        System.out.println(EmpiricalCDF.distance(empiricalCDFs.get(0),empiricalCDFs.get(2)));
        System.out.println(EmpiricalCDF.distance(empiricalCDFs.get(1),empiricalCDFs.get(2)));
    }

}