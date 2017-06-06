package edu.neu.ccs.pyramid.core.regression;

import com.fasterxml.jackson.databind.ObjectMapper;
import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.regression.regression_tree.TreeRule;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;
import edu.neu.ccs.pyramid.core.dataset.RegDataSet;
import edu.neu.ccs.pyramid.core.dataset.StandardFormat;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.core.regression.regression_tree.RegressionTree;
import org.apache.commons.lang3.time.StopWatch;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class RulesTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test1();
    }

    static void test1() throws Exception{
        int numLeaves = 4;

        RegDataSet dataSet = StandardFormat.loadRegDataSet("/Users/chengli/Datasets/slice_location/standard/featureList.txt",
                "/Users/chengli/Datasets/slice_location/standard/labels.txt", ",", DataSetType.REG_DENSE, false);
        System.out.println(dataSet.isDense());




        int[] activeFeatures = IntStream.range(0, dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();
        RegTreeConfig regTreeConfig = new RegTreeConfig();


        regTreeConfig.setMaxNumLeaves(numLeaves);
        regTreeConfig.setMinDataPerLeaf(5);


        regTreeConfig.setNumSplitIntervals(100);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        RegressionTree regressionTree = RegTreeTrainer.fit(regTreeConfig, dataSet);

        TreeRule rule1 = new TreeRule(regressionTree,dataSet.getRow(100));
        TreeRule rule2 = new TreeRule(regressionTree,dataSet.getRow(1));
        ConstantRule rule3 = new ConstantRule(0.8);
        Rule rule4 = new LinearRule();
        List<Rule> rules = new ArrayList<>();
        rules.add(rule1);
        rules.add(rule2);
        rules.add(rule3);
        rules.add(rule4);
        ObjectMapper mapper = new ObjectMapper();
        mapper.writeValue(new File(TMP,"decision.json"), rules);
    }

}