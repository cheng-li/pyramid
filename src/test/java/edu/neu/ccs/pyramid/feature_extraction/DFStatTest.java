package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;

import static org.junit.Assert.*;

public class DFStatTest {
    public static void main(String[] args) throws Exception{
        test1();
    }

    private static void test1() throws Exception{
        SingleLabelIndex index = new SingleLabelIndex.Builder().setIndexName("cnn").setClientType("node")
                .setLabelField("label").build();
        String[] ids = new String[1000];
        for (int i=0;i<1000;i++){
            ids[i] = ""+i;
        }
        DFStat dfStat = new DFStat(9,"barack",index,ids);
        System.out.println(dfStat);
        index.close();
    }

}