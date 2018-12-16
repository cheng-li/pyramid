package edu.neu.ccs.pyramid.regression.ls_logistic_boost;


import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.RegDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostOptimizer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;

import java.util.Arrays;

public class LSLogisticBoostTest {
    public static void main(String[] args) throws Exception{
        test3();
    }

    private static void test1(){
        RegDataSet regDataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(10000).numFeatures(2)
                .build();

        for (int i=0;i<regDataSet.getNumDataPoints();i++){
            double r = Math.random();
            double s = Math.random();
            regDataSet.setFeatureValue(i,0,r);
            regDataSet.setFeatureValue(i,0,s);
            if (r+s<0.1){
                regDataSet.setLabel(i,2);
            } else {
                regDataSet.setLabel(i,-2);
            }
        }

        LSLogisticBoost lsLogisticBoost = new LSLogisticBoost();
        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(10);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSLogisticBoostOptimizer optimizer = new LSLogisticBoostOptimizer(lsLogisticBoost,regDataSet,regTreeFactory);
        optimizer.setShrinkage(100);
        optimizer.initialize();
        for (int i=0;i<100;i++){
            System.out.println("training rmse = "+ RMSE.rmse(lsLogisticBoost,regDataSet));
            optimizer.iterate();
        }

        for (int i=0;i<1000;i++){
            System.out.println(regDataSet.getLabels()[i]+" "+lsLogisticBoost.predict(regDataSet.getRow(i)));
        }

//        System.out.println("********************** LSBOOST **************");
//        LSBoost lsBoost = new LSBoost();
//        LSBoostOptimizer lsBoostOptimizer = new LSBoostOptimizer(lsBoost,regDataSet,regTreeFactory);
//        lsBoostOptimizer.initialize();
//        for (int i=0;i<100;i++){
//            System.out.println("training rmse = "+ RMSE.rmse(lsBoost,regDataSet));
//            lsBoostOptimizer.iterate();
//        }
//
//        for (int i=0;i<1000;i++){
//            System.out.println(regDataSet.getLabels()[i]+" "+lsBoost.predict(regDataSet.getRow(i)));
//        }
    }


    private static void test2() throws Exception{
        RegDataSet train = TRECFormat.loadRegDataSet("/Users/chengli/Downloads/spam/train", DataSetType.REG_DENSE,true);
        RegDataSet test = TRECFormat.loadRegDataSet("/Users/chengli/Downloads/spam/test", DataSetType.REG_DENSE,true);

        for (int i=0;i<train.getNumDataPoints();i++){
            if (train.getLabels()[i]==0){
                train.setLabel(i,-1);
            } else {
                train.setLabel(i,2);
            }
        }

        for (int i=0;i<test.getNumDataPoints();i++){
            if (test.getLabels()[i]==0){
                test.setLabel(i,-1);
            } else {
                test.setLabel(i,2);
            }
        }

        LSLogisticBoost lsLogisticBoost = new LSLogisticBoost();
        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(5);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSLogisticBoostOptimizer optimizer = new LSLogisticBoostOptimizer(lsLogisticBoost,train,regTreeFactory);
        optimizer.initialize();
        for (int i=0;i<100;i++){
            System.out.println("training rmse = "+ RMSE.rmse(lsLogisticBoost,train));
            System.out.println("test rmse = "+ RMSE.rmse(lsLogisticBoost,test));
            optimizer.iterate();
        }

        for (int i=0;i<test.getNumDataPoints();i++){
            System.out.println(test.getLabels()[i]+" "+lsLogisticBoost.predict(test.getRow(i)));
        }
//        System.out.println(Arrays.toString(lsLogisticBoost.predict(train)));

        LSBoost lsBoost = new LSBoost();
        LSBoostOptimizer lsBoostOptimizer = new LSBoostOptimizer(lsBoost, train, regTreeFactory);

        System.out.println("LSBOOST");
        lsBoostOptimizer.initialize();
        for (int i=0;i<100;i++){
            System.out.println("training rmse = "+ RMSE.rmse(lsBoost,train));
            System.out.println("test rmse = "+ RMSE.rmse(lsBoost,test));
            lsBoostOptimizer.iterate();
        }
    }

    private static void test3() throws Exception{
        RegDataSet train = TRECFormat.loadRegDataSet("/Users/chengli/Downloads/spam/train", DataSetType.REG_DENSE,true);
        RegDataSet test = TRECFormat.loadRegDataSet("/Users/chengli/Downloads/spam/test", DataSetType.REG_DENSE,true);


        LSLogisticBoost lsLogisticBoost = new LSLogisticBoost();
        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(10);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSLogisticBoostOptimizer optimizer = new LSLogisticBoostOptimizer(lsLogisticBoost,train,regTreeFactory);
        optimizer.setShrinkage(1);
        optimizer.initialize();
        for (int i=0;i<200;i++){
            System.out.println("training rmse = "+ RMSE.rmse(lsLogisticBoost,train));
            System.out.println("test rmse = "+ RMSE.rmse(lsLogisticBoost,test));
            optimizer.iterate();
        }

        for (int i=0;i<test.getNumDataPoints();i++){
            System.out.println(test.getLabels()[i]+" "+lsLogisticBoost.predict(test.getRow(i)));
        }

        System.out.println("********************** LSBOOST **************");
        LSBoost lsBoost = new LSBoost();
        LSBoostOptimizer lsBoostOptimizer = new LSBoostOptimizer(lsBoost,train,regTreeFactory);
        lsBoostOptimizer.setShrinkage(0.1);
        lsBoostOptimizer.initialize();
        for (int i=0;i<100;i++){
            System.out.println("training rmse = "+ RMSE.rmse(lsBoost,train));
            System.out.println("test rmse = "+ RMSE.rmse(lsBoost,test));
            lsBoostOptimizer.iterate();
        }

        for (int i=0;i<test.getNumDataPoints();i++){
            System.out.println(test.getLabels()[i]+" "+lsBoost.predict(test.getRow(i)));
        }
    }

}