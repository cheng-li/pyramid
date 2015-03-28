package edu.neu.ccs.pyramid.elasticsearch;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.spans.SpanNearQuery;
import org.apache.lucene.search.spans.SpanQuery;
import org.apache.lucene.search.spans.SpanTermQuery;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.common.xcontent.ToXContent;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.elasticsearch.index.query.*;

import static org.junit.Assert.*;

public class ESIndexTest {
    public static void main(String[] args) throws Exception{
        test9();

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

    static void test3() throws Exception{
        ESIndex index = new ESIndex.Builder().setClientType("node").setIndexName("cnn")
                .build();
        System.out.println(index.getDocs("possibl").size());
        index.close();
    }

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
        System.out.println(index.getClient().prepareGet().setIndex("ohsumed_20000").setType("document").setFields("codes").setId("0")
                .execute().actionGet().getField("codes").getValues());
        System.out.println(index.getClient().prepareGet().setIndex("ohsumed_20000").setType("document").setFields("real_labels").setId("0")
                .execute().actionGet().getField("real_labels").getValues());
//        builder.endObject();
//        System.out.println(builder.prettyPrint().string());
        System.out.println(index.getClient().prepareGet().setIndex("ohsumed_20000").setType("document").setFields("codes").setId("0")
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

        QueryBuilder builder = QueryBuilders.spanNearQuery().clause(new SpanTermQueryBuilder("body", "skip"))
                .clause(new SpanTermQueryBuilder("body", "thi"))
                .clause(new SpanTermQueryBuilder("body","movi")).slop(0)
                .inOrder(true);


        SearchResponse response = index.getClient().prepareSearch(index.getIndexName()).setSize(index.getNumDocs()).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(builder).
                execute().actionGet();
        System.out.println(response.getHits().getTotalHits());


        index.close();
    }

}