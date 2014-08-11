package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DenseRegDataSet;
import edu.neu.ccs.pyramid.dataset.FeatureSetting;
import edu.neu.ccs.pyramid.dataset.SparseRegDataSet;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.feature.FeatureType;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static org.junit.Assert.*;

public class RegressionTreeTest {


    public static void main(String[] args) throws Exception{
        test5();

    }

    static void test1() throws Exception{
        int numLeaves = 2;

        SparseRegDataSet dataSet = new SparseRegDataSet(5,1);
        dataSet.setFeatureValue(0,0,1.5);
        dataSet.setFeatureValue(1,0,1);
        dataSet.setFeatureValue(2,0,1.6);
        dataSet.setFeatureValue(3,0,4);
        dataSet.setFeatureValue(4,0,2);
        dataSet.setLabel(0,1);
        dataSet.setLabel(1,2);
        dataSet.setLabel(2,3);
        dataSet.setLabel(3,4);
        dataSet.setLabel(4,1);
        for (int i=0;i<dataSet.getNumFeatures();i++){
            FeatureSetting setting = new FeatureSetting();
            setting.setFeatureType(FeatureType.NUMERICAL);
            dataSet.putFeatureSetting(i,setting);
        }
        double[] labels = dataSet.getLabels();
        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setActiveFeatures(activeFeatures);
        regTreeConfig.setDataSet(dataSet);
        regTreeConfig.setLabels(labels);
        regTreeConfig.setMaxNumLeaves(numLeaves);
        regTreeConfig.setMinDataPerLeaf(1);
        regTreeConfig.setActiveDataPoints(activeDataPoints);
        regTreeConfig.useDefaultOutputCalculator();
        regTreeConfig.setNumSplitIntervals(4);

        RegressionTree regressionTree = new RegressionTree();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        regressionTree.fit(regTreeConfig);

        System.out.println(stopWatch);

        double mseValue = MSE.mse(regressionTree, dataSet);
        System.out.println(mseValue);
//        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );
    }
    
    /**
     * slice location data
     */
    static void test2() throws Exception {
        int numLeaves = 70;

        SparseRegDataSet dataSet = SparseRegDataSet.loadStandard(new File("/Users/chengli/Datasets/slice_location/standard/features.txt"),
                new File("/Users/chengli/Datasets/slice_location/standard/labels.txt"), ",");

        for (int i=0;i<dataSet.getNumFeatures();i++){
            FeatureSetting setting = new FeatureSetting();
            setting.setFeatureType(FeatureType.NUMERICAL);
            dataSet.putFeatureSetting(i,setting);
        }


        double[] labels = dataSet.getLabels();

        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setActiveFeatures(activeFeatures);
        regTreeConfig.setDataSet(dataSet);
        regTreeConfig.setLabels(labels);
        regTreeConfig.setMaxNumLeaves(numLeaves);
        regTreeConfig.setMinDataPerLeaf(1);
        regTreeConfig.setActiveDataPoints(activeDataPoints);
        regTreeConfig.useDefaultOutputCalculator();
        regTreeConfig.setNumSplitIntervals(4);

        RegressionTree regressionTree = new RegressionTree();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        regressionTree.fit(regTreeConfig);

        System.out.println(stopWatch);

        double mseValue = MSE.mse(regressionTree, dataSet);
        System.out.println(mseValue);
//        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );

    }

    /**
     * slice location data, one column
     */
    static void test3() throws Exception {
        int numLeaves = 2;
        int featureIndex = 19;

        SparseRegDataSet dataSet1 = SparseRegDataSet.loadStandard(new File("/Users/chengli/Datasets/slice_location/standard/features.txt"),
                new File("/Users/chengli/Datasets/slice_location/standard/labels.txt"), ",");
        SparseRegDataSet dataSet = new SparseRegDataSet(dataSet1.getNumDataPoints(),1);

        for (int i=0;i<dataSet.getNumDataPoints();i++){
            dataSet.setFeatureValue(i,0,dataSet1.getFeatureColumn(featureIndex).getVector().get(i));
            dataSet.setLabel(i,dataSet1.getLabels()[i]);
        }



        for (int i=0;i<dataSet.getNumFeatures();i++){
            FeatureSetting setting = new FeatureSetting();
            setting.setFeatureType(FeatureType.NUMERICAL);
            dataSet.putFeatureSetting(i,setting);
        }
        System.out.println(dataSet.getFeatureColumn(0).getVector());
        System.out.println("max="+dataSet.getFeatureColumn(0).getVector().maxValue());


        double[] labels = dataSet.getLabels();
        System.out.println("my label sum = "+ Arrays.stream(labels).sum());
        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setActiveFeatures(activeFeatures);
        regTreeConfig.setDataSet(dataSet);
        regTreeConfig.setLabels(labels);
        regTreeConfig.setMaxNumLeaves(numLeaves);
        regTreeConfig.setMinDataPerLeaf(1);
        regTreeConfig.setActiveDataPoints(activeDataPoints);
        regTreeConfig.useDefaultOutputCalculator();
        regTreeConfig.setNumSplitIntervals(50);

        RegressionTree regressionTree = new RegressionTree();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        regressionTree.fit(regTreeConfig);

        System.out.println(stopWatch);

//        double mseValue = MSE.mse(regressionTree, dataSet);
//        System.out.println(mseValue);
        System.out.println(regressionTree);
        System.out.println(regressionTree.getRootReduction()    );


//        double threshold = 0.9927031171464085;
        double threshold = -0.22514464020729064;
        Vector vector = dataSet.getFeatureColumn(0).getVector();
        double leftSum=0;
        double rightSum=0;
        int leftcount=0;
        int rightcount=0;
        double sumlabel=0;
        for (int i=0;i<vector.size();i++){
            sumlabel+=labels[i];
            if (vector.get(i)<=threshold){
                leftSum += labels[i];
                leftcount += 1;
            } else {
                rightSum += labels[i];
                rightcount += 1;
            }
        }
        System.out.println("leftsum"+leftSum);
        System.out.println("rightsum"+rightSum);
        System.out.println("leftcount"+leftcount);
        System.out.println("rightcount"+rightcount);
        System.out.println("maxlrNormalizedSquareSum "+leftSum*leftSum/leftcount + rightSum*rightSum/rightcount);
        double myreduction = leftSum*leftSum/leftcount + rightSum*rightSum/rightcount - sumlabel*sumlabel/vector.size();
        System.out.println("my reduction="+myreduction);
    }

    static void test4() throws Exception{
        int numLeaves = 2;

        SparseRegDataSet dataSet = new SparseRegDataSet(5,1);
        dataSet.setFeatureValue(0,0,-1.5);
        dataSet.setFeatureValue(1,0,-1);
        dataSet.setFeatureValue(2,0,-1.6);
        dataSet.setFeatureValue(3,0,-4);
        dataSet.setFeatureValue(4,0,-2);
        dataSet.setLabel(0,1);
        dataSet.setLabel(1,2);
        dataSet.setLabel(2,3);
        dataSet.setLabel(3,4);
        dataSet.setLabel(4,1);
        for (int i=0;i<dataSet.getNumFeatures();i++){
            FeatureSetting setting = new FeatureSetting();
            setting.setFeatureType(FeatureType.NUMERICAL);
            dataSet.putFeatureSetting(i,setting);
        }
        double[] labels = dataSet.getLabels();
        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setActiveFeatures(activeFeatures);
        regTreeConfig.setDataSet(dataSet);
        regTreeConfig.setLabels(labels);
        regTreeConfig.setMaxNumLeaves(numLeaves);
        regTreeConfig.setMinDataPerLeaf(1);
        regTreeConfig.setActiveDataPoints(activeDataPoints);
        regTreeConfig.useDefaultOutputCalculator();
        regTreeConfig.setNumSplitIntervals(50);

        RegressionTree regressionTree = new RegressionTree();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        regressionTree.fit(regTreeConfig);

        System.out.println(stopWatch);

        double mseValue = MSE.mse(regressionTree, dataSet);
        System.out.println(mseValue);
        System.out.println(regressionTree);
        System.out.println(regressionTree.getRootReduction()    );
    }


    /**
     * slice location data, dense
     */
    static void test5() throws Exception {
        int numLeaves = 70;

        DenseRegDataSet dataSet = DenseRegDataSet.loadStandard(new File("/Users/chengli/Datasets/slice_location/standard/features.txt"),
                new File("/Users/chengli/Datasets/slice_location/standard/labels.txt"), ",");

        for (int i=0;i<dataSet.getNumFeatures();i++){
            FeatureSetting setting = new FeatureSetting();
            setting.setFeatureType(FeatureType.NUMERICAL);
            dataSet.putFeatureSetting(i,setting);
        }


        double[] labels = dataSet.getLabels();

        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setActiveFeatures(activeFeatures);
        regTreeConfig.setDataSet(dataSet);
        regTreeConfig.setLabels(labels);
        regTreeConfig.setMaxNumLeaves(numLeaves);
        regTreeConfig.setMinDataPerLeaf(1);
        regTreeConfig.setActiveDataPoints(activeDataPoints);
        regTreeConfig.useDefaultOutputCalculator();
        regTreeConfig.setNumSplitIntervals(100);

        RegressionTree regressionTree = new RegressionTree();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        regressionTree.fit(regTreeConfig);

        System.out.println(stopWatch);

        double mseValue = MSE.mse(regressionTree, dataSet);
        System.out.println(mseValue);
//        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );

    }

//    static void test6(){
//        RegTreeConfig regTreeConfig = new RegTreeConfig();
//        regTreeConfig.setNumSplitIntervals(4);
//        double[] features = {1,1,2,2};
//        double[] labels = {3,4,5,6};
//        System.out.println(IntervalSplitter.generateIntervals(regTreeConfig,features,labels));
//
//    }

//    static void test7(){
//        RegTreeConfig regTreeConfig = new RegTreeConfig();
//        regTreeConfig.setNumSplitIntervals(4);
//        double[] features = {1,1,2,2,3,4};
//        double[] labels = {3,4,5,6,7,8};
//        System.out.println(IntervalSplitter.generateIntervals(regTreeConfig,features,labels));
//
//    }

//    static void test8(){
//        RegTreeConfig regTreeConfig = new RegTreeConfig();
//        regTreeConfig.setNumSplitIntervals(10);
//        double[] features = {1,1,2,2,3,4};
//        double[] labels = {3,4,5,6,7,8};
//        List<Interval> intervals = IntervalSplitter.generateIntervals(regTreeConfig, features, labels);
//        System.out.println(intervals);
//        List<Interval> compressed = IntervalSplitter.compress(intervals);
//        System.out.println(compressed);
//
//    }
}