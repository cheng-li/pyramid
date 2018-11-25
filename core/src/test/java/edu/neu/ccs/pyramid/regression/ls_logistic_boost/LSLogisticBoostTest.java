package edu.neu.ccs.pyramid.regression.ls_logistic_boost;


import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostOptimizer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;

import java.util.Arrays;

public class LSLogisticBoostTest {
    public static void main(String[] args) throws Exception{
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

//        LSBoost lsBoost = new LSBoost();
//        LSBoostOptimizer lsBoostOptimizer = new LSBoostOptimizer(lsBoost, train, regTreeFactory);
//
//        System.out.println("LSBOOST");
//        lsBoostOptimizer.initialize();
//        for (int i=0;i<100;i++){
//            System.out.println("training rmse = "+ RMSE.rmse(lsBoost,train));
//            System.out.println("test rmse = "+ RMSE.rmse(lsBoost,test));
//            lsBoostOptimizer.iterate();
//        }
    }

}