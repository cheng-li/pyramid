package edu.neu.ccs.pyramid.elasticsearch;

import static org.junit.Assert.*;

public class MultiLabelIndexTest {
    public static void main(String[] args) throws Exception{
//        test1();
        test2();
    }

    private static void test1() throws Exception{
        MultiLabelIndex index = new MultiLabelIndex.Builder()
                .setIndexName("ohsumed_20000")
                .setBodyField("body")
                .setClientType("node")
                .setClusterName("elasticsearch")
                .setDocumentType("document")
                .setExtMultiLabelField("real_labels")
                .build();
        System.out.println(index.getNumDocs());
        index.close();
    }

    private static void test2() throws Exception{
        MultiLabelIndex index = new MultiLabelIndex.Builder()
                .setIndexName("ohsumed_20000")
                .setBodyField("body")
                .setClientType("node")
                .setClusterName("elasticsearch")
                .setDocumentType("document")
                .setExtMultiLabelField("real_labels")
                .build();
        System.out.println(index.DFForClass("period","Bacterial Infections and Mycoses"));
        index.close();
    }



}