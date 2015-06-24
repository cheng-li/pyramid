package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.elasticsearch.MultiLabelIndex;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;

import java.io.File;
import java.util.List;

import static org.junit.Assert.*;

public class DFStatsTest {
    private static final Config config = new Config("configs/local.config");
    private static final String DATASETS = config.getString("input.datasets");
    private static final String TMP = config.getString("output.tmp");

    public static void main(String[] args) throws Exception{
//        test4();
//        test5();
        test6();

    }

    static void test1() throws Exception{
        DFStats dfStats = new DFStats(9);
        System.out.println(dfStats);

        SingleLabelIndex index = new SingleLabelIndex.Builder().setIndexName("cnn").setClientType("node")
                .setLabelField("label").build();
        dfStats.updateByOneDoc(index,"0");
        System.out.println(dfStats);
        index.close();

    }

    static void test2() throws Exception{
        DFStats dfStats = new DFStats(9);
        System.out.println(dfStats);
        SingleLabelIndex index = new SingleLabelIndex.Builder().setIndexName("cnn").setClientType("node")
                .setLabelField("label").build();
        dfStats.update(index);
        System.out.println(dfStats);
        index.close();

    }

    static void test3() throws Exception{
        DFStats dfStats = new DFStats(9);
//        System.out.println(dfStats);
        SingleLabelIndex index = new SingleLabelIndex.Builder().setIndexName("cnn").setClientType("node")
                .setLabelField("label").build();
        dfStats.update(index);
//        System.out.println(dfStats);
        index.close();

        dfStats.sort();
        dfStats.serialize("/Users/chengli/tmp/dfstats.ser");


    }

    static void test4() throws Exception{

        DFStats dfStats = DFStats.deserialize("/Users/chengli/tmp/dfstats.ser");
        List<DFStat> list = dfStats.getSortedDFStats(0,20);
        for (int i=0;i<Math.min(100,list.size());i++){
            System.out.println(list.get(i).getPhrase());
        }

    }

//    private static void test5() throws Exception{
//        MultiLabelIndex index = new MultiLabelIndex.Builder()
//                .setIndexName("ohsumed_20000")
//                .setBodyField("body")
//                .setClientType("node")
//                .setClusterName("elasticsearch")
//                .setDocumentType("document")
//                .setExtMultiLabelField("real_labels")
//                .build();
//        DFStats dfStats = new DFStats(23);
////
//        dfStats.update(index,labelTranslator);
//        dfStats.sort();
//        dfStats.serialize(new File(TMP,"dfstats.ser"));
//        index.close();
//    }

    static void test6() throws Exception{

        DFStats dfStats = DFStats.deserialize(new File(TMP,"dfstats.ser"));
        List<DFStat> list = dfStats.getSortedDFStats(0,20);
        for (int i=0;i<Math.min(100,list.size());i++){
            System.out.println(list.get(i).getPhrase());
        }

    }



}