package edu.neu.ccs.pyramid.regression;

import edu.neu.ccs.pyramid.util.*;
import junit.framework.TestCase;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class IsotonicRegressionTest  {
    public static void main(String[] args) throws Exception{
        test7();
    }

    private static void test1(){
        double[] numbers = new double[10];
        for (int i=0;i<numbers.length;i++){
            numbers[i] = i;
        }
        System.out.println(new IsotonicRegression(numbers));
    }

    private static void test2(){
        double[] numbers = {9,3,2,4,3,3,5,9};
        for (int i=0;i<numbers.length;i++){
            numbers[i] = i;
        }
        System.out.println(new IsotonicRegression(numbers));
    }

    private static void test3(){
        double[] numbers = new double[10];
        for (int i=0;i<numbers.length;i++){
            numbers[i] = Math.random();
        }
        System.out.println(new IsotonicRegression(numbers));
    }

    private static void test4() throws Exception{
        double[] locations = new double[50];
        double[] numbers = new double[50];
        for (int i=0;i<numbers.length;i++){
            locations[i] = Sampling.doubleUniform(-Math.PI/2, Math.PI/2);
            numbers[i] = Math.sin(locations[i])+Sampling.doubleUniform(-0.2,0.2);
        }
        IsotonicRegression isotonicRegression = new IsotonicRegression(locations, numbers,false);
        List<Double> grid =Grid.uniform(-Math.PI/2, Math.PI/2,1000);
        double[] gridValues = grid.stream().mapToDouble(g->isotonicRegression.predict(g)).toArray();


        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_locations"), PrintUtil.toSimpleString(locations));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_numbers"), PrintUtil.toSimpleString(numbers));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_locations"), PrintUtil.toSimpleString(isotonicRegression.getLocations()));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_numbers"), PrintUtil.toSimpleString(isotonicRegression.getValues()));

        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_grid"), ListUtil.toSimpleString(grid));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_gridvalues"), PrintUtil.toSimpleString(gridValues));
    }

    private static void test5() throws Exception{
        double[] locations = new double[50];
        double[] numbers = new double[50];
        for (int i=0;i<numbers.length;i++){
            locations[i] = Sampling.doubleUniform(-2, 2);
            double p = 1/(1+Math.exp(-locations[i]));
            if (Math.random()>p){
                numbers[i]=0;
            } else {
                numbers[i]=1;
            }
        }
        IsotonicRegression isotonicRegression = new IsotonicRegression(locations, numbers,false);
        List<Double> grid =Grid.uniform(-2, 2,1000);
        double[] gridValues = grid.stream().mapToDouble(g->isotonicRegression.predict(g)).toArray();


        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_locations"), PrintUtil.toSimpleString(locations));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_numbers"), PrintUtil.toSimpleString(numbers));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_locations"), PrintUtil.toSimpleString(isotonicRegression.getLocations()));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_numbers"), PrintUtil.toSimpleString(isotonicRegression.getValues()));

        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_grid"), ListUtil.toSimpleString(grid));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_gridvalues"), PrintUtil.toSimpleString(gridValues));
    }


    private static void test6() throws Exception{
        double[] locations = new double[5000];
        double[] numbers = new double[5000];
        for (int i=0;i<numbers.length;i++){
            locations[i] = Sampling.doubleUniform(0, 1);
            if (Math.random()>locations[i]){
                numbers[i]=0;
            } else {
                numbers[i]=1;
            }
        }
        IsotonicRegression isotonicRegression = new IsotonicRegression(locations, numbers,false);
        List<Double> grid =Grid.uniform(0, 1,1000);
        double[] gridValues = grid.stream().mapToDouble(g->isotonicRegression.predict(g)).toArray();


        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_locations"), PrintUtil.toSimpleString(locations));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_numbers"), PrintUtil.toSimpleString(numbers));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_locations"), PrintUtil.toSimpleString(isotonicRegression.getLocations()));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_numbers"), PrintUtil.toSimpleString(isotonicRegression.getValues()));

        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_grid"), ListUtil.toSimpleString(grid));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_gridvalues"), PrintUtil.toSimpleString(gridValues));
    }


    private static void test7() throws Exception{
        double[] locations = new double[50];
        double[] numbers = new double[50];
        for (int i=0;i<numbers.length;i++){
            locations[i] = Sampling.doubleUniform(0, 1);
            if (Math.random()>locations[i]){
                numbers[i]=0;
            } else {
                numbers[i]=1;
            }
        }

        List<Pair<Double, Double>> list = new ArrayList<>();
        for (int i=0;i<locations.length;i++){
            list.add(new Pair<>(locations[i],numbers[i]));
        }

        list = MathUtil.mergeEveryK(list,5);
        double[] locationsMerged = list.stream().mapToDouble(p->p.getFirst()).toArray();
        double[] numbersMerged = list.stream().mapToDouble(p->p.getSecond()).toArray();


        IsotonicRegression isotonicRegression = new IsotonicRegression(locationsMerged, numbersMerged,false);
        List<Double> grid =Grid.uniform(0, 1,1000);
        double[] gridValues = grid.stream().mapToDouble(g->isotonicRegression.predict(g)).toArray();


        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_locations"), PrintUtil.toSimpleString(locations));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_numbers"), PrintUtil.toSimpleString(numbers));

        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_locations_merged"), PrintUtil.toSimpleString(locationsMerged));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/input_numbers_merged"), PrintUtil.toSimpleString(numbersMerged));

        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_locations"), PrintUtil.toSimpleString(isotonicRegression.getLocations()));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_numbers"), PrintUtil.toSimpleString(isotonicRegression.getValues()));

        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_grid"), ListUtil.toSimpleString(grid));
        FileUtils.writeStringToFile(new File("/Users/chengli/tmp/iso/output_gridvalues"), PrintUtil.toSimpleString(gridValues));
    }

}