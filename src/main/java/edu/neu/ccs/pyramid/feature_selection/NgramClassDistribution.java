package edu.neu.ccs.pyramid.feature_selection;

import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.feature.Ngram;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.IdsFilterBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.index.query.SpanNearQueryBuilder;
import org.elasticsearch.index.query.SpanTermQueryBuilder;
import org.elasticsearch.search.aggregations.bucket.terms.Terms;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collection;

import static org.elasticsearch.search.aggregations.AggregationBuilders.terms;

/**
 * Created by chengli on 4/25/15.
 */
public class NgramClassDistribution implements Serializable{
    private static final long serialVersionUID = 1L;
    private Ngram ngram;
    private long totalCount;
    private long[] classCounts;

    public NgramClassDistribution(Ngram ngram, int numClasses) {
        this.ngram = ngram;
        this.classCounts = new long[numClasses];
    }

    public NgramClassDistribution(Ngram ngram, ESIndex index,
                                  String labelField, String[] ids,
                                  LabelTranslator labelTranslator) {
        this.ngram = ngram;
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

    public Ngram getNgram() {
        return ngram;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("NgramClassDistribution{");
        sb.append("ngram=").append(ngram);
        sb.append(", totalCount=").append(totalCount);
        sb.append(", classCounts=").append(Arrays.toString(classCounts));
        sb.append('}');
        return sb.toString();
    }
}
