package edu.neu.ccs.pyramid.classification.l2boost;

import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBConfig;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTBTrainer;
import edu.neu.ccs.pyramid.classification.boosting.lktb.LKTreeBoost;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.regression.regression_tree.LeafOutputType;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.List;

import static org.junit.Assert.*;

public class L2BoostTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    static void test1() throws Exception{
        buildTest();
        loadTest();
    }

    static void buildTest() throws Exception {

        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());

        L2Boost boost = new L2Boost();

        RegTreeConfig regTreeConfig = new RegTreeConfig()
                .setMaxNumLeaves(7);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        regTreeFactory.setLeafOutputCalculator(new L2BLeafOutputCalculator());

        L2BoostOptimizer optimizer = new L2BoostOptimizer(boost, dataSet, regTreeFactory);
        optimizer.setShrinkage(0.1);
        optimizer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        for (int round = 0; round < 200; round++) {
            System.out.println("round=" + round);
            optimizer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(boost, dataSet);
        System.out.println("accuracy=" + accuracy);

        Serialization.serialize(boost,new File(TMP,"boost"));
    }

    static void loadTest() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.CLF_SPARSE, true);
        L2Boost boost = (L2Boost)Serialization.deserialize(new File(TMP,"boost"));
        double accuracy = Accuracy.accuracy(boost, dataSet);
        System.out.println("accuracy=" + accuracy);

    }

    static void test2() throws Exception {


        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/spam/trec_data/train.trec"),
                DataSetType.CLF_SPARSE, true);
        System.out.println(dataSet.getMetaInfo());

        L2Boost boost = new L2Boost();

        L2BoostOptimizer optimizer = new L2BoostOptimizer(boost, dataSet);
        optimizer.setShrinkage(1);
        optimizer.initialize();

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        for (int round = 0; round < 200; round++) {
            System.out.println("round=" + round);
            optimizer.iterate();
        }
        stopWatch.stop();
        System.out.println(stopWatch);


        double accuracy = Accuracy.accuracy(boost, dataSet);
        System.out.println("accuracy=" + accuracy);

    }

}