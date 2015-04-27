package edu.neu.ccs.pyramid.feature_selection;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.feature.SpanNotNgram;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.*;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;

import static org.elasticsearch.search.aggregations.AggregationBuilders.terms;

/**
 * Created by chengli on 4/25/15.
 */
public class FeatureDistribution implements Serializable{
    private static final long serialVersionUID = 2L;
    private Feature feature;
    private long totalCount;
    private long[] classCounts;



    public FeatureDistribution(Ngram ngram, ESIndex index,
                               String labelField, String[] ids,
                               LabelTranslator labelTranslator) {
        this.feature = ngram;
        int numClasses = labelTranslator.getNumClasses();
        this.classCounts = new long[numClasses];

        String field = ngram.getField();
        int slop = ngram.getSlop();
        boolean inOrder = ngram.isInOrder();
        SpanNearQueryBuilder queryBuilder = QueryBuilders.spanNearQuery();
        for (String term: ngram.getTerms()){
            queryBuilder.clause(new SpanTermQueryBuilder(field, term));
        }
        queryBuilder.inOrder(inOrder);
        queryBuilder.slop(slop);

        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(index.getDocumentType());
        idsFilterBuilder.addIds(ids);

        SearchResponse response = index.getClient().prepareSearch(index.getIndexName()).setSize(0).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(queryBuilder, idsFilterBuilder))
                .addAggregation(terms("agg").field(labelField).size(Integer.MAX_VALUE))
                .execute().actionGet();

        this.totalCount = response.getHits().getTotalHits();

        Terms terms = response.getAggregations().get("agg");
        Collection<Terms.Bucket> buckets = terms.getBuckets();

        for (Terms.Bucket bucket: buckets){
            String extLabel = bucket.getKey();
            long count = bucket.getDocCount();
            int classIndex = labelTranslator.toIntLabel(extLabel);
            this.classCounts[classIndex] = count;
        }
    }

    public FeatureDistribution(SpanNotNgram ngram, ESIndex index,
                               String labelField, String[] ids,
                               LabelTranslator labelTranslator) {
        this.feature = ngram;
        int numClasses = labelTranslator.getNumClasses();
        this.classCounts = new long[numClasses];

        Ngram include = ngram.getInclude();
        String field1 = include.getField();
        int slop1 = include.getSlop();
        boolean inOrder1 = include.isInOrder();
        SpanNearQueryBuilder queryBuilder1 = QueryBuilders.spanNearQuery();
        for (String term: include.getTerms()){
            queryBuilder1.clause(new SpanTermQueryBuilder(field1, term));
        }
        queryBuilder1.inOrder(inOrder1);
        queryBuilder1.slop(slop1);


        Ngram exclude = ngram.getExclude();

        String field2 = exclude.getField();
        int slop2 = exclude.getSlop();
        boolean inOrder2 = exclude.isInOrder();
        SpanNearQueryBuilder queryBuilder2 = QueryBuilders.spanNearQuery();
        for (String term: exclude.getTerms()){
            queryBuilder2.clause(new SpanTermQueryBuilder(field2, term));
        }
        queryBuilder2.inOrder(inOrder2);
        queryBuilder2.slop(slop2);

        int pre = ngram.getPre();
        int post = ngram.getPost();

        SpanNotQueryBuilder spanNotQueryBuilder = QueryBuilders.spanNotQuery().include(queryBuilder1)
                .exclude(queryBuilder2);
        //todo upgrade to 1.5
//                .pre(pre).post(post);
        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(index.getDocumentType());
        idsFilterBuilder.addIds(ids);

        SearchResponse response = index.getClient().prepareSearch(index.getIndexName()).setSize(0).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(spanNotQueryBuilder, idsFilterBuilder))
                .addAggregation(terms("agg").field(labelField).size(Integer.MAX_VALUE))
                .execute().actionGet();


        this.totalCount = response.getHits().getTotalHits();

        Terms terms = response.getAggregations().get("agg");
        Collection<Terms.Bucket> buckets = terms.getBuckets();

        for (Terms.Bucket bucket: buckets){
            String extLabel = bucket.getKey();
            long count = bucket.getDocCount();
            int classIndex = labelTranslator.toIntLabel(extLabel);
            this.classCounts[classIndex] = count;
        }
    }

    public void setClassCount(int classIndex, long count){
        classCounts[classIndex] = count;
    }

    public long getClassCount(int classIndex){
        return classCounts[classIndex];
    }

    public long getTotalCount() {
        return totalCount;
    }

    public void setTotalCount(long totalCount) {
        this.totalCount = totalCount;
    }

    public Feature getFeature() {
        return feature;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("NgramClassDistribution{");
        sb.append("feature=").append(feature);
        sb.append(", totalCount=").append(totalCount);
        sb.append(", classCounts=").append(Arrays.toString(classCounts));
        sb.append('}');
        return sb.toString();
    }
}
