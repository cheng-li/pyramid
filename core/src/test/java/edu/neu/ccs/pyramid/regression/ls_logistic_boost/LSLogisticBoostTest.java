package edu.neu.ccs.pyramid.regression.ls_logistic_boost;


import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;

public class LSLogisticBoostTest {
    public static void main(String[] args) throws Exception{
        RegDataSet train = TRECFormat.loadRegDataSet("/Users/chengli/Downloads/spam/train", DataSetType.REG_DENSE,true);
        RegDataSet test = TRECFormat.loadRegDataSet("/Users/chengli/Downloads/spam/test", DataSetType.REG_DENSE,true);
        LSLogisticBoost lsLogisticBoost = new LSLogisticBoost();
        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(5);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSLogisticBoostOptimizer optimizer = new LSLogisticBoostOptimizer(lsLogisticBoost,train,regTreeFactory);
        optimizer.initialize();
        for (int i=0;i<1000;i++){
            System.out.println("training rmse = "+ RMSE.rmse(lsLogisticBoost,train));
            System.out.println("test rmse = "+ RMSE.rmse(lsLogisticBoost,test));
            optimizer.iterate();
        }
    }

}