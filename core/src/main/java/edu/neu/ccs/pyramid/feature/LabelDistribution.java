package edu.neu.ccs.pyramid.feature;

import static org.elasticsearch.search.aggregations.AggregationBuilders.terms;

import java.util.Collection;

import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.IdsQueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;

/**
 * Created by chengli on 4/30/15.
 */
public class LabelDistribution {

    public static long[] getLabelDistribution(ESIndex index,
                               String labelField, String[] ids,
                               LabelTranslator labelTranslator) {

        int numClasses = labelTranslator.getNumClasses();
        long[] distribution = new long[numClasses];


        IdsQueryBuilder idsFilterBuilder = new IdsQueryBuilder(index.getDocumentType());
        idsFilterBuilder.addIds(ids);

        SearchResponse response = index.getClient().prepareSearch(index.getIndexName()).setSize(0).
                setTrackScores(false).
                setFetchSource(false).setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.
                		boolQuery()
                		.must(idsFilterBuilder)
                		.should(QueryBuilders.matchAllQuery()))
                .addAggregation(terms("agg").field(labelField).size(Integer.MAX_VALUE))
                .execute().actionGet();


        Terms terms = response.getAggregations().get("agg");
        Collection<Terms.Bucket> buckets = terms.getBuckets();

        for (Terms.Bucket bucket: buckets){
            String extLabel = (String) bucket.getKey();
            long count = bucket.getDocCount();
            int classIndex = labelTranslator.toIntLabel(extLabel);
            distribution[classIndex] = count;
        }

        return distribution;
    }
}
