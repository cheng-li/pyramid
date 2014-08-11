package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.DenseRegDataSet;
import edu.neu.ccs.pyramid.dataset.FeatureSetting;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.feature.FeatureType;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.stream.IntStream;

import static org.junit.Assert.*;

public class RegTreeTrainerTest {

    public static void main(String[] args) throws Exception{
        test3();

    }

    static void test1(){
        PriorityQueue<Node> leaves = new PriorityQueue<>(Comparator.comparing(Node::getReduction).reversed());
        leaves.add(new Node().setReduction(8));
        leaves.add(new Node().setReduction(1));
        leaves.add(new Node().setReduction(3));
        leaves.add(new Node().setReduction(2));
        leaves.add(new Node().setReduction(4));
        leaves.add(new Node().setReduction(5));
        leaves.add(new Node().setReduction(1.7));
        System.out.println(leaves.poll().getReduction());
        System.out.println(leaves.poll().getReduction());
        System.out.println(leaves.poll().getReduction());
        System.out.println(leaves.poll().getReduction());

    }

    /**
     * slice location data, dense
     */
    static void test2() throws Exception {
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

        regTreeConfig.setMaxNumLeaves(numLeaves);
        regTreeConfig.setMinDataPerLeaf(5);
        regTreeConfig.setActiveDataPoints(activeDataPoints);

        regTreeConfig.setNumSplitIntervals(100);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig,dataSet,labels);


        System.out.println(stopWatch);

        double mseValue = MSE.mse(regressionTree, dataSet);
        System.out.println(mseValue);
//        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );

    }

    /**
     * slice location data, dense
     */
    static void test3() throws Exception {
        int numLeaves = 70;

        DenseRegDataSet dataSet = DenseRegDataSet.loadStandard(new File("/Users/chengli/Datasets/slice_location/standard/features.txt"),
                new File("/Users/chengli/Datasets/slice_location/standard/labels.txt"), ",");

        for (int i=0;i<dataSet.getNumFeatures();i++){
            FeatureSetting setting = new FeatureSetting();
            setting.setFeatureType(FeatureType.NUMERICAL);
            dataSet.putFeatureSetting(i,setting);
        }



        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setActiveFeatures(activeFeatures);

        regTreeConfig.setMaxNumLeaves(numLeaves);
        regTreeConfig.setMinDataPerLeaf(5);
        regTreeConfig.setActiveDataPoints(activeDataPoints);

        regTreeConfig.setNumSplitIntervals(100);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig,dataSet);


        System.out.println(stopWatch);

        double mseValue = MSE.mse(regressionTree, dataSet);
        System.out.println(mseValue);
//        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );

    }

}