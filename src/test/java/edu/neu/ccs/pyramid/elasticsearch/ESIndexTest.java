package edu.neu.ccs.pyramid.elasticsearch;

import static org.junit.Assert.*;

public class ESIndexTest {
    public static void main(String[] args) throws Exception{
        test7();

    }

    static void test1() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getNumDocs());
        index.close();
    }

    static void test2() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getTerms("0"));
        index.close();
    }

    static void test3() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getDocs("possibl").size());
        index.close();
    }

    static void test4() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getLabel("0"));
        System.out.println(index.getExtLabel("0"));
        System.out.println(index.getStringField("0","split"));

        index.close();
    }

    static void test5() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getTermStats("0"));


        index.close();
    }

    static void test6() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getTermVector("0"));


        index.close();
    }

    static void test7() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.phraseDF(index.getBodyField(),"its mission",0));
        System.out.println(index.phraseDF(index.getBodyField(),"it mission",0));
        index.close();
    }

}