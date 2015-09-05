package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.DataSetUtil;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.feature.FeatureUtility;
import edu.neu.ccs.pyramid.feature_selection.FusedKolmogorovFilter;

import java.io.File;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * test fused kolmogorov filter on a complete matrix
 * Created by chengli on 3/26/15.
 */
public class Exp77 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("Please specify a properties file.");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        List<Integer> selected = selection(config);
        dump(config,selected);
    }

    private static List<Integer> selection(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        FusedKolmogorovFilter filter = new FusedKolmogorovFilter();
        List<FeatureUtility> featureUtilityList = IntStream.range(0, dataSet.getNumFeatures()).parallel()
                .mapToObj(j -> {
                    FeatureUtility featureUtility = new FeatureUtility(dataSet.getFeatureList().get(j));
                    double score = filter.score(dataSet.getColumn(j), dataSet.getLabels(), dataSet.getNumClasses());
                    featureUtility.setUtility(score);
                    return featureUtility;
                })
                .sorted(Comparator.comparing(FeatureUtility::getUtility).reversed()).limit(config.getInt("limit"))
                .collect(Collectors.toList());
        for (FeatureUtility featureUtility: featureUtilityList){
            System.out.println(featureUtility);
        }
        return featureUtilityList.stream().map(utility -> utility.getFeature().getIndex()).collect(Collectors.toList());
    }

    private static void dump(Config config, List<Integer> selected)throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet trimed = DataSetUtil.sampleFeatures(dataSet, selected);
        TRECFormat.save(trimed,new File(config.getString("output.folder"),"train.trec"));

        ClfDataSet testSet = TRECFormat.loadClfDataSet(new File(input, "test.trec"),
                DataSetType.CLF_SPARSE, true);
        ClfDataSet trimedTest = DataSetUtil.sampleFeatures(testSet, selected);
        TRECFormat.save(trimedTest,new File(config.getString("output.folder"),"test.trec"));
    }
}
