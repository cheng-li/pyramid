package edu.neu.ccs.pyramid.core.regression.regression_tree;

import edu.neu.ccs.pyramid.core.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.core.configuration.Config;
import edu.neu.ccs.pyramid.core.dataset.TRECFormat;
import edu.neu.ccs.pyramid.core.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.core.dataset.DataSetType;

import java.io.File;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SplitterTest {
    private static final Config config = new Config("config/local.properties");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(DATASETS, "/imdb/train.trec"),
                DataSetType.CLF_SPARSE, true);
        PriorProbClassifier priorProbClassifier = new PriorProbClassifier(dataSet.getNumClasses());
        priorProbClassifier.fit(dataSet);

        double[] gradient = priorProbClassifier.getGradient(dataSet,1);
        RegTreeConfig regTreeConfig = new RegTreeConfig();
        int[] activeFeatures = IntStream.range(0,dataSet.getNumFeatures()).toArray();
        int[] activeDataPoints = IntStream.range(0,dataSet.getNumDataPoints()).toArray();

        Comparator<SplitResult> comparator = Comparator.comparing(SplitResult::getReduction);
        List<Integer> results = Splitter.getAllSplits(regTreeConfig,dataSet,gradient)
                .stream().sorted(comparator.reversed()).map(result -> result.getFeatureIndex()).limit(100)
                .collect(Collectors.toList());
//        results.stream().forEach(i-> System.out.println(dataSet.getFeatureSetting(i).getFeatureName()));
        System.out.println(results);
    }

}