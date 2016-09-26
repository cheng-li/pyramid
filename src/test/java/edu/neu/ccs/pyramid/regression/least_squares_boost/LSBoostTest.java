package edu.neu.ccs.pyramid.regression.least_squares_boost;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.StandardFormat;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.MSE;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.simulation.RegressionSynthesizer;

import java.io.File;
import java.util.stream.IntStream;

public class LSBoostTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }



    private static void test1() throws Exception{
        RegDataSet trainSet = TRECFormat.loadRegDataSet(new File(DATASETS,"abalone/folds/fold_1/train.trec"),DataSetType.REG_DENSE,true);

        RegDataSet testSet = TRECFormat.loadRegDataSet(new File(DATASETS,"abalone/folds/fold_1/test.trec"),DataSetType.REG_DENSE,true);


        LSBoost lsBoost = new LSBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(3);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSBoostOptimizer optimizer = new LSBoostOptimizer(lsBoost, trainSet, regTreeFactory);
        optimizer.setShrinkage(0.1);
        optimizer.initialize();

        for (int i=0;i<100;i++){
            System.out.println("iteration "+i);
            System.out.println("train RMSE = "+ RMSE.rmse(lsBoost, trainSet));
            System.out.println("test RMSE = "+ RMSE.rmse(lsBoost, testSet));
            optimizer.iterate();
        }
    }

}