package edu.neu.ccs.pyramid.core.feature_selection;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import edu.neu.ccs.pyramid.core.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.core.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.core.feature.Feature;
import edu.neu.ccs.pyramid.core.feature.Ngram;
import edu.neu.ccs.pyramid.core.feature.SpanNotNgram;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.*;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.elasticsearch.search.aggregations.AggregationBuilders.terms;

/**
 * Created by chengli on 4/25/15.
 */
@JsonSerialize(using = FeatureDistribution.Serializer.class)
public class FeatureDistribution implements Serializable{
    private static final long serialVersionUID = 3L;
    private Feature feature;
    private long totalCount;
    private long[] occurInEach;
    private long[] labelDistribution;
    private LabelTranslator labelTranslator;



    public FeatureDistribution(Ngram ngram, ESIndex index,
                               String labelField, String[] ids,
                               LabelTranslator labelTranslator,
                               long[] labelDistribution) {
        this.feature = ngram;
        this.labelDistribution = labelDistribution;
        this.labelTranslator = labelTranslator;
        int numClasses = labelTranslator.getNumClasses();
        this.occurInEach = new long[numClasses];

        String field = ngram.getField();
        int slop = ngram.getSlop();
        boolean inOrder = ngram.isInOrder();
        SpanNearQueryBuilder queryBuilder = QueryBuilders.spanNearQuery(new SpanTermQueryBuilder(field, ngram.getTerms()[0]), slop);
        for (int i  = 1; i < ngram.getTerms().length; i++){
            queryBuilder.addClause(new SpanTermQueryBuilder(field, ngram.getTerms()[i]));
        }
        queryBuilder.inOrder(inOrder);

        IdsQueryBuilder idsFilterBuilder = new IdsQueryBuilder(index.getDocumentType());
        idsFilterBuilder.addIds(ids);

        SearchResponse response = index.getClient().prepareSearch(index.getIndexName()).setSize(0).
                setTrackScores(false).
                setFetchSource(false).setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.boolQuery()
                					  .must(idsFilterBuilder)
                					  .should(queryBuilder))
                .addAggregation(terms("agg").field(labelField).size(Integer.MAX_VALUE))
                .execute().actionGet();

        this.totalCount = response.getHits().getTotalHits();

        Terms terms = response.getAggregations().get("agg");
        Collection<Terms.Bucket> buckets = terms.getBuckets();

        for (Terms.Bucket bucket: buckets){
            String extLabel = (String) bucket.getKey();
            long count = bucket.getDocCount();
            int classIndex = labelTranslator.toIntLabel(extLabel);
            this.occurInEach[classIndex] = count;
        }
    }

    public FeatureDistribution(SpanNotNgram ngram, ESIndex index,
                               String labelField, String[] ids,
                               LabelTranslator labelTranslator) {
        this.feature = ngram;
        int numClasses = labelTranslator.getNumClasses();
        this.occurInEach = new long[numClasses];

        Ngram include = ngram.getInclude();
        String field1 = include.getField();
        int slop1 = include.getSlop();
        boolean inOrder1 = include.isInOrder();
        SpanNearQueryBuilder queryBuilder1 = QueryBuilders.spanNearQuery(new SpanTermQueryBuilder(field1, include.getTerms()[0]), slop1);
        for (int i = 1; i < include.getTerms().length; i++){
            queryBuilder1.addClause(new SpanTermQueryBuilder(field1, include.getTerms()[i]));
        }
        queryBuilder1.inOrder(inOrder1);


        Ngram exclude = ngram.getExclude();

        String field2 = exclude.getField();
        int slop2 = exclude.getSlop();
        boolean inOrder2 = exclude.isInOrder();
        SpanNearQueryBuilder queryBuilder2 = QueryBuilders.spanNearQuery(new SpanTermQueryBuilder(field2, exclude.getTerms()[0]), slop2);
        for (int i = 1; i < exclude.getTerms().length; i++){
            queryBuilder2.addClause(new SpanTermQueryBuilder(field2, exclude.getTerms()[i]));
        }
        queryBuilder2.inOrder(inOrder2);

        int pre = ngram.getPre();
        int post = ngram.getPost();

        SpanNotQueryBuilder spanNotQueryBuilder = new SpanNotQueryBuilder(queryBuilder1, queryBuilder2);
        //todo upgrade to 1.5
//                .pre(pre).post(post);
        IdsQueryBuilder idsFilterBuilder = new IdsQueryBuilder(index.getDocumentType());
        idsFilterBuilder.addIds(ids);

        SearchResponse response = index.getClient().prepareSearch(index.getIndexName()).setSize(0).
                setTrackScores(false).
                setFetchSource(false).setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders
                		.boolQuery()
                		.must(idsFilterBuilder)
                		.should(spanNotQueryBuilder))
                .addAggregation(terms("agg").field(labelField).size(Integer.MAX_VALUE))
                .execute().actionGet();


        this.totalCount = response.getHits().getTotalHits();

        Terms terms = response.getAggregations().get("agg");
        Collection<Terms.Bucket> buckets = terms.getBuckets();

        for (Terms.Bucket bucket: buckets){
            String extLabel = (String) bucket.getKey();
            long count = bucket.getDocCount();
            int classIndex = labelTranslator.toIntLabel(extLabel);
            this.occurInEach[classIndex] = count;
        }
    }

    public void setClassCount(int classIndex, long count){
        occurInEach[classIndex] = count;
    }

    public long getClassCount(int classIndex){
        return occurInEach[classIndex];
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

    public void setFeature(Feature feature) {
        this.feature = feature;
    }

    public long[] getOccurInEach() {
        return occurInEach;
    }

    public void setOccurInEach(long[] occurInEach) {
        this.occurInEach = occurInEach;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("NgramClassDistribution{");
        sb.append("feature=").append(feature);
        sb.append(", totalCount=").append(totalCount);
        sb.append(", occurInEach=").append(Arrays.toString(occurInEach));
        sb.append('}');
        return sb.toString();
    }

    public List<String> pretty(){
        List<String> list = new ArrayList<>();
        for (int i=0;i<occurInEach.length;i++){
            StringBuilder sb = new StringBuilder();
            sb.append(labelTranslator.toExtLabel(i)).append(":");
            sb.append(occurInEach[i]).append("/").append(labelDistribution[i]);
            list.add(sb.toString());
        }
        return list;
    }

    public static class Serializer extends JsonSerializer<FeatureDistribution> {
        @Override
        public void serialize(FeatureDistribution distribution, JsonGenerator jsonGenerator,
                              SerializerProvider serializerProvider) throws IOException, JsonProcessingException {
            jsonGenerator.writeStartObject();
            jsonGenerator.writeObjectField("feature",distribution.feature);
            jsonGenerator.writeNumberField("totalCount", distribution.totalCount);
            jsonGenerator.writeObjectField("occurrence",distribution.pretty());
            jsonGenerator.writeEndObject();

        }
    }
}
