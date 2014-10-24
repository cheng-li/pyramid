package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.MSE;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.stream.IntStream;

public class RegTreeTrainerTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test2();
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
        System.out.println(dataSet.getNumDataPoints());
        System.out.println(dataSet.getNumFeatures());


        double[] labels = dataSet.getLabels();

        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0, dataSet.getNumDataPoints()).toArray();
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
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            System.out.println(RegTreeInspector.getMatchedLeaf(regressionTree,dataSet.getFeatureRow(i)));
        }
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
                "/Users/chengli/Datasets/slice_location/standard/labels.txt", ",", DataSetType.REG_SPARSE);
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
        RegDataSet dataSet = new SparseRegDataSet(50000,5000);
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
                DataSetType.REG_DENSE,false);

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

    /**
     * spam, dense
     */
    static void test7() throws Exception {
        int numLeaves = 10;

        RegDataSet dataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/train_data.txt"),
                new File(DATASETS, "spam/train_label.txt"), ",", DataSetType.REG_DENSE);

        for (int i=0;i<dataSet.getNumFeatures();i++){
            dataSet.getFeatureColumn(i).getSetting().setFeatureType(FeatureType.NUMERICAL);
        }
        System.out.println(dataSet.getNumDataPoints());
        System.out.println(dataSet.getNumFeatures());


        double[] labels = dataSet.getLabels();

        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0, dataSet.getNumDataPoints()).toArray();
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
        System.out.println("training mse="+mseValue);

        RegDataSet testDataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/test_data.txt"),
                new File(DATASETS, "spam/test_label.txt"), ",", DataSetType.REG_DENSE);
        System.out.println("test mse="+MSE.mse(regressionTree,testDataSet));

//        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );

    }

    /**
     * spam, dense, with random missing values
     */
    static void test8() throws Exception {
        int numLeaves = 10;

        RegDataSet dataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/train_data.txt"),
                new File(DATASETS, "spam/train_label.txt"), ",", DataSetType.REG_DENSE);

        for (int i=0;i<dataSet.getNumFeatures();i++){
            dataSet.getFeatureColumn(i).getSetting().setFeatureType(FeatureType.NUMERICAL);
        }

        for (int i=0;i<dataSet.getNumDataPoints();i++){
            for (int j=0;j<dataSet.getNumFeatures();j++){
                if (Math.random()<0.1){
                    dataSet.setFeatureValue(i,j,Double.NaN);
                }
            }
        }

        double[] labels = dataSet.getLabels();

        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0, dataSet.getNumDataPoints()).toArray();
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
        System.out.println("training mse="+mseValue);

        RegDataSet testDataSet = StandardFormat.loadRegDataSet(new File(DATASETS, "spam/test_data.txt"),
                new File(DATASETS, "spam/test_label.txt"), ",", DataSetType.REG_DENSE);

        for (int i=0;i<testDataSet.getNumDataPoints();i++){
            for (int j=0;j<testDataSet.getNumFeatures();j++){
                if (Math.random()<0.1){
                    testDataSet.setFeatureValue(i,j,Double.NaN);
                }
            }
        }

        System.out.println("test mse="+MSE.mse(regressionTree,testDataSet));

//        System.out.println(regressionTree);
//        System.out.println(regressionTree.getRootReduction()    );

    }

}