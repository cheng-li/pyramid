package edu.neu.ccs.pyramid.ranking;


import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.RegDataSet;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.eval.NDCG;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class LambdaMARTOptimizerTest  {
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        RegDataSet train = TRECFormat.loadRegDataSet("/Users/chengli/Downloads/spam/train", DataSetType.REG_DENSE,true);
        train = DataSetUtil.shuffleRows(train,0);
        RegDataSet test = TRECFormat.loadRegDataSet("/Users/chengli/Downloads/spam/test", DataSetType.REG_DENSE,true);
        test = DataSetUtil.shuffleRows(test,0);
        LambdaMART lambdaMART = new LambdaMART();
        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(5);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        List<List<Integer>> instancesInQuery = new ArrayList<>();
        List<Integer> all = IntStream.range(0, train.getNumDataPoints()).boxed().collect(Collectors.toList());
        instancesInQuery.add(all);

        LambdaMARTOptimizer optimizer = new LambdaMARTOptimizer(lambdaMART,train, train.getLabels(),regTreeFactory, instancesInQuery);

        optimizer.initialize();
        for (int i=0;i<500;i++){
            System.out.println("==================================");
            System.out.println("iter "+i+"");
            System.out.println("ndcg = "+ NDCG.ndcg(test.getLabels(),lambdaMART.predict(test)));
//            for (int j=0;j<train.getNumDataPoints();j++){
//                System.out.println("label = "+train.getLabels()[j]+" pred = "+lambdaMART.predict(train.getRow(j)));
//            }
            optimizer.iterate();

        }
    }

}