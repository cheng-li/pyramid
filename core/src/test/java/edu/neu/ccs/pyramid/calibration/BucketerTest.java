package edu.neu.ccs.pyramid.calibration;


import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.regression.IsotonicRegression;
import edu.neu.ccs.pyramid.util.Pair;

import java.util.Arrays;
import java.util.stream.IntStream;

public class BucketerTest {
    public static void main(String[] args) throws Exception {
        test1();
    }


    private static void test1(){
        double[] x = {0.2,0.1,0.3,0.6,0.4,0.5,0.8,0.7,0.9};
        double[] y = {2,1,3,6,4,5,8,7,9};
        System.out.println(Bucketer.groupWithEqualSize(x,y,3));
        System.out.println(Bucketer.groupWithEqualSize(x,y,4));
        System.out.println(Bucketer.groupWithEqualSize(x,y,2));
    }


    private static void test2() throws Exception{
        RegDataSet dataSet = TRECFormat.loadRegDataSet("/Users/chengli/tmp/iso/LRout7/dump3", DataSetType.REG_SPARSE,true);
        double[] x = new double[dataSet.getNumDataPoints()];
        double[] y = new double[dataSet.getNumDataPoints()];
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            x[i] = dataSet.getRow(i).get(1);
            y[i] = dataSet.getLabels()[i];
        }

        Bucketer.Result result  = Bucketer.groupWithEqualSize(x,y,1000);
        System.out.println(result);
        IsotonicRegression isotonicRegression = new IsotonicRegression(x,y);
        double[] prediction = Arrays.stream(result.averageX).map(isotonicRegression::predict).toArray();
        System.out.println("prediction");
        System.out.println(Arrays.toString(prediction));

        String dis = Displayer.displayCalibrationResult(IntStream.range(0,dataSet.getNumDataPoints()).mapToObj(i-> new Pair<>(isotonicRegression.predict(dataSet.getRow(i).get(1)),dataSet.getLabels()[i])));
        System.out.println(dis);
    }



}