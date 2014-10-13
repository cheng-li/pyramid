package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;

import java.util.List;

import static org.junit.Assert.*;

public class DFStatsTest {
    public static void main(String[] args) throws Exception{
        test4();

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

}