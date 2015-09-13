package edu.neu.ccs.pyramid.multilabel_classification;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.util.SetUtil;

import java.io.File;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.Assert.*;

public class MultiLabelSuggesterTest {
    private static final Config config = new Config("config/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");
    public static void main(String[] args) throws Exception{
        test4();
    }

    private static void test1()throws Exception{
         MultiLabelClfDataSet dataSet  = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                 DataSetType.ML_CLF_DENSE, true);
        MultiLabelSuggester suggester = new MultiLabelSuggester(dataSet,2);
        System.out.println("bmm="+suggester.getBmm());
        System.out.println("new labels = "+suggester.suggestNewOnes(100));
    }

    private static void test2()throws Exception{
        MultiLabelClfDataSet dataSet  = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.ML_CLF_DENSE, true);
        MultiLabelSuggester suggester = new MultiLabelSuggester(dataSet,1);
        System.out.println("bmm="+suggester.getBmm());
        System.out.println("new labels = "+suggester.suggestNewOnes(100));
    }

    private static void test3()throws Exception{
        MultiLabelClfDataSet dataSet  = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "/spam/trec_data/test.trec"),
                DataSetType.ML_CLF_DENSE, true);
        MultiLabelSuggester suggester = new MultiLabelSuggester(dataSet,30);
        System.out.println("bmm="+suggester.getBmm());
        System.out.println("new labels = "+suggester.suggestNewOnes(100));
    }

    private static void test4()throws Exception{
        MultiLabelClfDataSet dataSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/train.trec"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelClfDataSet testSet = TRECFormat.loadMultiLabelClfDataSet(new File(DATASETS, "ohsumed/3/test.trec"), DataSetType.ML_CLF_SPARSE, true);
        MultiLabelSuggester suggester = new MultiLabelSuggester(dataSet,10);
        System.out.println("bmm="+suggester.getBmm());


        Set<MultiLabel> trainLabels = Arrays.stream(dataSet.getMultiLabels()).collect(Collectors.toSet());
        Set<MultiLabel> testLabels = Arrays.stream(testSet.getMultiLabels()).collect(Collectors.toSet());
        Set<MultiLabel> newintest = SetUtil.complement(testLabels,trainLabels);
        System.out.println("new labels in test set = "+newintest);

        Set<MultiLabel> sampled = suggester.suggestNewOnes(1000);
        System.out.println("sampled:");
        for (MultiLabel multiLabel: sampled){
            System.out.println(multiLabel+"\t"+newintest.contains(multiLabel));
        }


    }


}