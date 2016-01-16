package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

/**
 * Created by chengli on 1/1/16.
 */
public class Exp140 {
    public static void main(String[] args) throws Exception{
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);

        System.out.println(config);
        MultiLabelClfDataSet trainSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"), DataSetType.ML_CLF_SPARSE, true);
        List<MultiLabel> list = DataSetUtil.gatherMultiLabels(trainSet);
        Set<MultiLabel> set = new HashSet<>();
        set.addAll(list);
        System.out.println("train combinations = "+set.size());

        long count = IntStream.range(0,testSet.getNumDataPoints()).filter(i->set.contains(testSet.getMultiLabels()[i])).count();
        System.out.println(count*1.0/testSet.getNumDataPoints());


    }
}
