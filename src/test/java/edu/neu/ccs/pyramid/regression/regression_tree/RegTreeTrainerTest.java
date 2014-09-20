package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.dataset.FeatureType;
import org.apache.commons.lang3.time.StopWatch;

import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.stream.IntStream;

public class RegTreeTrainerTest {

    public static void main(String[] args) throws Exception{
        test6();
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

        RegDataSet dataSet = StandardFormat.loadRegDataSet("/Users/chengli/Datasets/slice_location/standard/features.txt",
                "/Users/chengli/Datasets/slice_location/standard/labels.txt", ",", DataSetType.REG_DENSE);

        for (int i=0;i<dataSet.getNumFeatures();i++){
            dataSet.getFeatureColumn(i).getSetting().setFeatureType(FeatureType.NUMERICAL);
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
        int numLeaves = 4;

        RegDataSet dataSet = StandardFormat.loadRegDataSet("/Users/chengli/Datasets/slice_location/standard/features.txt",
                "/Users/chengli/Datasets/slice_location/standard/labels.txt", ",", DataSetType.REG_DENSE);
        System.out.println(dataSet.isDense());
        for (int i=0;i<dataSet.getNumFeatures();i++){
            dataSet.getFeatureColumn(i).getSetting().setFeatureType(FeatureType.NUMERICAL);
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
        System.out.println(regressionTree.getNumLeaves());
        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );

    }

    /**
     * slice location data, sparse
     */
    static void test4() throws Exception {
        int numLeaves = 4;

        RegDataSet dataSet = StandardFormat.loadRegDataSet("/Users/chengli/Datasets/slice_location/standard/features.txt",
                "/Users/chengli/Datasets/slice_location/standard/labels.txt", ",", DataSetType.RANK_SPARSE);
        System.out.println(dataSet.isDense());
        for (int i=0;i<dataSet.getNumFeatures();i++){
            dataSet.getFeatureColumn(i).getSetting().setFeatureType(FeatureType.NUMERICAL);
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
        System.out.println(regressionTree.getNumLeaves());
//        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );

    }

    /**
     * random sparse data, sparse matrix
     */
    static void test5() throws Exception {
        int numLeaves = 4;
//        RegDataSet dataSet = new DenseRegDataSet(50000,10000);
        RegDataSet dataSet = new SparseRegDataSet(500,500);
        IntStream.range(0,dataSet.getNumDataPoints())
                .forEach(i-> IntStream.range(0,dataSet.getNumFeatures())
                .forEach(j-> {
                    boolean set = Math.random()<0.01;
                    if (set){
                        dataSet.setFeatureValue(i,j,Math.random());
                    }
                    dataSet.setLabel(i,Math.random());
                }));
        System.out.println("created");


        for (int i=0;i<dataSet.getNumFeatures();i++){
            dataSet.getFeatureColumn(i).getSetting().setFeatureType(FeatureType.NUMERICAL);
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
        System.out.println(regressionTree.getNumLeaves());
//        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );

    }

    /**
     * housing
     * @throws Exception
     */
    static void test6() throws Exception{

        RegDataSet dataSet = TRECFormat.loadRegDataSet("/Users/chengli/Datasets/housing/trec_format/all.trec",
                DataSetType.REG_DENSE,true);

        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        regTreeConfig.setActiveFeatures(activeFeatures);
        int numLeaves = 1000;
        regTreeConfig.setMaxNumLeaves(numLeaves);
        regTreeConfig.setMinDataPerLeaf(5);
        regTreeConfig.setActiveDataPoints(activeDataPoints);

        regTreeConfig.setNumSplitIntervals(100);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig,dataSet);
        System.out.println(regressionTree);
        double mseValue = MSE.mse(regressionTree, dataSet);
        System.out.println(mseValue);
        System.out.println(stopWatch);
        System.out.println(RegTreeInspector.featureImportance(regressionTree));

    }

}