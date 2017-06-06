package edu.neu.ccs.pyramid.core.elasticsearch;

import edu.neu.ccs.pyramid.core.feature.Ngram;
import edu.neu.ccs.pyramid.core.feature.SpanNotNgram;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.*;
import org.elasticsearch.search.SearchHit;

import java.util.ArrayList;
import java.util.List;

public class ESIndexTest {
    public static void main(String[] args) throws Exception{
        test22();

    }

    static void test1() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getNumDocs());
        index.close();
    }

    static void test2() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getTerms("0"));
        index.close();
    }

//    static void test3() throws Exception{
//        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("cnn")
//                .build();
//        System.out.println(index.termFilter("possibl").size());
//        index.close();
//    }

    static void test4() throws Exception{
        SingleLabelIndex index = new SingleLabelIndex.Builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getLabel("0"));
        System.out.println(index.getExtLabel("0"));
        System.out.println(index.getStringField("0","split"));

        index.close();
    }

    static void test5() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getTermStats("0"));


        index.close();
    }

    static void test6() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getTermVector("0"));


        index.close();
    }

    static void test7() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.phraseDF(index.getBodyField(),"its mission",0));
        System.out.println(index.phraseDF(index.getBodyField(),"it mission",0));
        index.close();
    }

    static void test8() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("ohsumed_20000")
                .build();
//        XContentBuilder builder = XContentFactory.jsonBuilder();
//        builder.startObject();
        System.out.println(index.getClient().prepareGet().setIndex("ohsumed_20000").setType("document").setStoredFields("codes").setId("0")
                .execute().actionGet().getField("codes").getValues());
        System.out.println(index.getClient().prepareGet().setIndex("ohsumed_20000").setType("document").setStoredFields("real_labels").setId("0")
                .execute().actionGet().getField("real_labels").getValues());
//        builder.endObject();
//        System.out.println(builder.prettyPrint().string());
        System.out.println(index.getClient().prepareGet().setIndex("ohsumed_20000").setType("document").setStoredFields("codes").setId("0")
                .execute().actionGet().getField("codes").getValues());
        index.close();
    }

    static void test9() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("ohsumed_20000")
                .build();
        System.out.println(index.getListField("0","codes"));
        System.out.println(index.getListField("0","real_labels"));
        System.out.println(index.getListField("0","file_name"));
        System.out.println(index.getStringListField("0","codes"));
        System.out.println(index.getStringListField("0","real_labels"));
        System.out.println(index.getStringListField("0","file_name"));
        index.close();
    }

    static void test10() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getField("0","a"));


        index.close();
    }

    static void test11() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("imdb")
                .build();
//        SpanTermQuery quick = new SpanTermQuery(new Term("body", "skip"));
//        SpanTermQuery brown = new SpanTermQuery(new Term("body", "thi"));
//        SpanTermQuery fox = new SpanTermQuery(new Term("body", "movi"));
//        SpanQuery[] quick_brown_dog =
//                new SpanQuery[]{quick, brown, fox};
//        SpanNearQuery snq =
//                new SpanNearQuery(quick_brown_dog, 0, true);

        QueryBuilder builder = QueryBuilders.spanNearQuery(new SpanTermQueryBuilder("body", "skip"), 0)
                .addClause(new SpanTermQueryBuilder("body", "thi"))
                .addClause(new SpanTermQueryBuilder("body","movi"))
                .inOrder(true);


        SearchResponse response = index.getClient().prepareSearch(index.getIndexName()).setSize(index.getNumDocs()).
                setTrackScores(false).
                setFetchSource(false).setExplain(false).setFetchSource(false).
                setQuery(builder).
                execute().actionGet();
        System.out.println(response.getHits().getTotalHits());


        index.close();
    }

    static void test12() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("imdb")
                .build();

        Ngram ngram1 = new Ngram();
        ngram1.setInOrder(true);
        ngram1.setNgram("recommend");
        ngram1.setField("body");
        ngram1.setSlop(0);


        Ngram ngram2 = new Ngram();
        ngram2.setInOrder(true);
        ngram2.setNgram("not");
        ngram2.setField("body");
        ngram2.setSlop(0);

        SpanNotNgram spanNotNgram = new SpanNotNgram();
        spanNotNgram.setInclude(ngram1);
        spanNotNgram.setExclude(ngram2);
        spanNotNgram.setPre(2);

        SearchResponse response = index.spanNot(spanNotNgram);

        System.out.println(response.getHits().getTotalHits());


        index.close();
    }


    static void test13() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("imdb")
                .build();

        System.out.println(index.getNumDocs());
        System.out.println(index.getAllDocs().size());
        index.close();
    }


//    static void test14() throws Exception{
//        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("imdb")
//                .build();
//
//        Ngram ngram1 = new Ngram();
//        ngram1.setInOrder(true);
//        ngram1.setNgram("really nice");
//        ngram1.setField("body");
//        ngram1.setSlop(0);
//
//        String[] ids = IntStream.range(0,500000).mapToObj(i-> ""+i).toArray(String[]::new);
//        SearchResponse searchResponse = index.spanNearFrequency(ngram1, ids);
//        System.out.println(searchResponse);
//
//        index.close();
//    }

    static void test15() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("imdb")
                .build();

//        String q = "{\"term\": {\"id\": 1}}";
//        String q = "{" +
//                "    \"match_all\": {}" +
//                "  }";
        String q = "{" +
                "    \"filtered\": {" +
                "      \"query\": {" +
                "        \"match_all\": {}" +
                "      }," +
                "      \"filter\": {" +
                "        \"term\": {" +
                "          \"split\": \"train\"" +
                "        }" +
                "      }" +
                "    }" +
                "  }";



        SearchResponse searchResponse = index.getClient().prepareSearch(index.getIndexName())
                .setSize(index.getNumDocs()).
                        setTrackScores(false).
                        setFetchSource(false).setExplain(false).setFetchSource(false)
                .setQuery(
                        QueryBuilders.wrapperQuery(q)).execute().actionGet();
        System.out.println(searchResponse.getHits().getTotalHits());

        index.close();
    }

    static void test16() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("imdb")
                .build();

//        String q = "{\"term\": {\"id\": 1}}";
//        String q = "{" +
//                "    \"match_all\": {}" +
//                "  }";
//        String q = "{" +
//                "    \"filtered\": {" +
//                "      \"query\": {" +
//                "        \"match_all\": {}" +
//                "      }," +
//                "      \"filter\": {" +
//                "        \"term\": {" +
//                "          \"split\": \"train\"" +
//                "        }" +
//                "      }" +
//                "    }" +
//                "  }";
        String q = "{\"filtered\":{\"query\":{\"match_all\":{}},\"filter\":{\"term\":{\"split\":\"train\"}}}}";



        List<String> res = index.matchStringQuery(q);
        System.out.println(res);

        index.close();
    }


    static void test17() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("imdb")
                .build();

        Ngram ngram = index.analyze("Story of a man who has unnatural feelings for a pig man test","my_analyzer");
        System.out.println(ngram);
        index.close();
    }


    static void test18() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("ohsumed_20000")
                .build();

//        String q = "{\"term\": {\"id\": 1}}";
//        String q = "{" +
//                "    \"match_all\": {}" +
//                "  }";
//        String q = "{" +
//                "    \"filtered\": {" +
//                "      \"query\": {" +
//                "        \"match_all\": {}" +
//                "      }," +
//                "      \"filter\": {" +
//                "        \"term\": {" +
//                "          \"split\": \"train\"" +
//                "        }" +
//                "      }" +
//                "    }" +
//                "  }";
        String q = "{\n" +
                "    \"bool\": {\n" +
                "      \"should\": [\n" +
                "        {\n" +
                "          \"constant_score\": {\n" +
                "            \"query\": {\n" +
                "              \"match\": {\n" +
                "                \"body\": \"repeated\"\n" +
                "              }\n" +
                "            }\n" +
                "          }\n" +
                "        },\n" +
                "                {\n" +
                "          \"constant_score\": {\n" +
                "            \"query\": {\n" +
                "              \"match\": {\n" +
                "                \"body\": \"cyclophosphamide\"\n" +
                "              }\n" +
                "            }\n" +
                "          }\n" +
                "        },\n" +
                "                        {\n" +
                "          \"constant_score\": {\n" +
                "            \"query\": {\n" +
                "              \"match\": {\n" +
                "                \"body\": \"cycles\"\n" +
                "              }\n" +
                "            }\n" +
                "          }\n" +
                "        },\n" +
                "                                {\n" +
                "          \"constant_score\": {\n" +
                "            \"query\": {\n" +
                "              \"match\": {\n" +
                "                \"body\": \"study\"\n" +
                "              }\n" +
                "            }\n" +
                "          }\n" +
                "        }\n" +
                "      ],\n" +
                "      \"minimum_should_match\": \"70%\"\n" +
                "    }\n" +
                "  }";



        SearchResponse  response = index.submitQuery(q);
        for (SearchHit searchHit : response.getHits()) {
            System.out.println(searchHit.getId()+" "+searchHit.getScore());
        }

        index.close();
    }

    static void test19() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("ohsumed_20000")
                .build();

        List<String> terms = new ArrayList<>();
        terms.add("repeated");
        terms.add("cyclophosphamide");
        terms.add("cycles");
        terms.add("study");
        SearchResponse response = index.minimumShouldMatch(terms, "body", 70);
        System.out.println(response.getHits().getTotalHits());
        for (SearchHit searchHit : response.getHits()) {
            System.out.println(searchHit.getId()+" "+searchHit.getScore());
        }
        index.close();
    }


//    static void test20() throws Exception{
//        try(ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("ohsumed_20000")
//                .build()){
//            List<String> terms = new ArrayList<>();
//            terms.add("repeated");
//            terms.add("cyclophosphamide");
//            terms.add("cycles");
//            terms.add("study");
//            String[] ids = {"AVYcLfPVDpWfZwAC_rp3", "AVYcLfbpDpWfZwAC_rt_"};
//            SearchResponse response = index.minimumShouldMatch(terms, "body", 70, ids);
//            System.out.println(response.getHits().getTotalHits());
//            for (SearchHit searchHit : response.getHits()) {
//                System.out.println(searchHit.getId()+" "+searchHit.getScore());
//            }
//            double a = 0/0;
//        }
//
//    }


    static void test21() throws Exception{
        try(ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("imdb")
                .build()){
            Ngram ngram1 = new Ngram();
            ngram1.setInOrder(true);
            ngram1.setNgram("really nice");
            ngram1.setField("body");
            ngram1.setSlop(0);
            String filterQuery = "{\"filtered\":{\"query\":{\"match_all\":{}},\"filter\":{\"term\":{\"split\":\"train\"}}}}";
            SearchResponse response = index.spanNear(ngram1, filterQuery, 10);
            for (SearchHit searchHit : response.getHits()) {
                System.out.println(searchHit.getId()+" "+searchHit.getScore());
            }
        }
    }


    static void test22() throws Exception{
        try(ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("imdb")
                .build()){
            List<String> terms = new ArrayList<>();
            terms.add("repeated");
            terms.add("cyclophosphamide");
            terms.add("cycles");
            terms.add("study");
            index.minimumShouldMatch(terms,"body", 70,10, null );
        }
    }

}