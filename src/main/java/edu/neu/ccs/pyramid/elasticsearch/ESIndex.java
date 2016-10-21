package edu.neu.ccs.pyramid.elasticsearch;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.feature.SpanNotNgram;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.DocsAndPositionsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.similarities.DefaultSimilarity;
import org.elasticsearch.action.admin.indices.analyze.AnalyzeRequest;
import org.elasticsearch.action.admin.indices.analyze.AnalyzeResponse;
import org.elasticsearch.action.admin.indices.mapping.get.GetFieldMappingsResponse;
import org.elasticsearch.action.admin.indices.mapping.get.GetMappingsResponse;
import org.elasticsearch.action.get.GetResponse;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.termvector.TermVectorResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.client.transport.TransportClient;
import org.elasticsearch.cluster.metadata.MappingMetaData;
import org.elasticsearch.common.lucene.search.function.CombineFunction;
import org.elasticsearch.common.settings.ImmutableSettings;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.transport.InetSocketTransportAddress;
import org.elasticsearch.index.query.*;
import org.elasticsearch.index.query.functionscore.ScoreFunctionBuilders;
import org.elasticsearch.node.Node;
import org.elasticsearch.search.SearchHit;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;
import static org.elasticsearch.search.aggregations.AggregationBuilders.terms;


/**
 * Created by chengli on 8/20/14.
 */
public class ESIndex implements AutoCloseable{
    public static final String STRING_MISSING_VALUE = "MISSING";

    private static final Logger logger = LogManager.getLogger();
    Client client;
    Node node;
    String indexName;
    int numDocs;
    String documentType;
    String clientType;
    String clusterName;
    String bodyField;

    //todo should have different caches for different fields
    /**
     * concurrent LRU cache for termvectors
     */
    LoadingCache<String,Map<Integer,String>> termVectorCache;


    public int getNumDocs() {
        return numDocs;
    }

    public List<String> getAllDocs(){
        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.matchAllQuery()).
                execute().actionGet();
        List<String> list = new ArrayList<>(response.getHits().getHits().length);
        for (SearchHit searchHit : response.getHits()) {
            list.add(searchHit.getId());
        }
        return list;
    }

    public Client getClient() {
        return client;
    }

    public String getIndexName() {
        return indexName;
    }



    public String getDocumentType() {
        return documentType;
    }

    public String getClientType() {
        return clientType;
    }

    public String getClusterName() {
        return clusterName;
    }

    public String getBodyField() {
        return bodyField;
    }

    /**
     *
     * @return terms stemmed
     */
    public Set<String> getTerms(String id) throws IOException {
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        TermVectorResponse response = client.prepareTermVector(indexName, documentType, id)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setSelectedFields(this.bodyField).
                        execute().actionGet();

        Terms terms = response.getFields().terms(this.bodyField);
        TermsEnum iterator = terms.iterator(null);
        Set<String> termsSet = new HashSet<>();
        for (int i=0;i<terms.size();i++){
            String term = iterator.next().utf8ToString();
           termsSet.add(term);
        }

        if(logger.isDebugEnabled()){
            logger.debug("time spent on getNgrams from doc "+id+" = "+stopWatch+
                    " It has "+termsSet.size()+" ngrams");
        }
        return termsSet;
    }


    /**
     * use as an inverted index
     * no score is computed
     * @param term stemmed term
     * @param ids
     * @return
     * @throws Exception
     */
    public List<String> getDocs(String term, String[] ids) throws Exception{
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }

        /**
         * setSize() has a huge impact on performance, the smaller the faster
         */

        //todo reuse idsFilterBuilder
        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);


        idsFilterBuilder.addIds(ids);

        TermFilterBuilder termFilterBuilder = new TermFilterBuilder(this.bodyField, term);

        SearchResponse response = client.prepareSearch(indexName).setSize(ids.length).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.constantScoreQuery(
                        FilterBuilders.andFilter(termFilterBuilder,
                                idsFilterBuilder))).
                execute().actionGet();
        List<String> list = new ArrayList<>(response.getHits().getHits().length);
        for (SearchHit searchHit : response.getHits()) {
            list.add(searchHit.getId());
        }
        if(logger.isDebugEnabled()){
            logger.debug("time spent on termFilter() for " + term+ " = " + stopWatch+
                    " There are "+list.size()+" matched docs");
        }
        return list;
    }

    /**
     * use as an inverted index
     * no score is computed
     * @param term stemmed term
     * @return
     * @throws Exception
     */
    public List<String> termFilter(String field, String term) throws Exception{
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }

        /**
         * setSize() has a huge impact on performance, the smaller the faster
         */

        TermFilterBuilder termFilterBuilder = new TermFilterBuilder(field, term);

        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.constantScoreQuery(
                        termFilterBuilder)).
                execute().actionGet();
        List<String> list = new ArrayList<>(response.getHits().getHits().length);
        for (SearchHit searchHit : response.getHits()) {
            list.add(searchHit.getId());
        }
        if(logger.isDebugEnabled()){
            logger.debug("time spent on termFilter() for " + term + " = " + stopWatch+
                    " There are "+list.size()+" matched docs");
        }
        return list;
    }

    public List<String> matchStringQuery(String query){
        SearchResponse response = client.prepareSearch(this.indexName)
                .setSize(numDocs).
                        setHighlighterFilter(false).setTrackScores(false).
                        setNoFields().setExplain(false).setFetchSource(false)
                .setQuery(QueryBuilders.wrapperQuery(query)).execute().actionGet();
        List<String> list = new ArrayList<>(response.getHits().getHits().length);
        for (SearchHit searchHit : response.getHits()) {
            list.add(searchHit.getId());
        }

        return list;
    }

    public SearchResponse submitQuery(String query){
        SearchResponse response = client.prepareSearch(this.indexName)
                .setSize(numDocs).
                        setHighlighterFilter(false).setTrackScores(false).
                        setNoFields().setExplain(false).setFetchSource(false)
                .setQuery(QueryBuilders.wrapperQuery(query)).execute().actionGet();
        return response;
    }


    public SearchResponse minimumShouldMatch(List<String> terms, String field, int percentage){
        StringBuilder sb = new StringBuilder();
        sb.append("{").append("\"bool\":{\"should\":[");
        for (int i=0;i<terms.size();i++){
            String term = terms.get(i);
            sb.append("{\"constant_score\": {\n" +
                    "            \"query\": {\n" +
                    "              \"match\": {");
            sb.append("\"").append(field).append("\":").append("\"").append(term).append("\"");
            sb.append("}\n" +
                    "            }\n" +
                    "          }\n" +
                    "        }");
            if (i!=terms.size()-1){
                sb.append(",");
            }

        }
        sb.append("],\"minimum_should_match\":");
        sb.append("\"").append(percentage).append("%").append("\"").append("}}");
        return submitQuery(sb.toString());
    }

    public SearchResponse minimumShouldMatch(String string, String field, int percentage, String analyzer){
        List<String> terms = analyzeString(string, analyzer);
        return minimumShouldMatch(terms, field, percentage);

    }

    public SearchResponse minimumShouldMatch(List<String> terms, String field, int percentage, String[] ids){
        StringBuilder sb = new StringBuilder();
        sb.append("{\n" +
                "    \"filtered\": {\n" +
                "      \"query\": {\n" +
                "        \"bool\": {");

        sb.append("\"should\":[");
        for (int i=0;i<terms.size();i++){
            String term = terms.get(i);
            sb.append("{\"constant_score\": {\n" +
                    "            \"query\": {\n" +
                    "              \"match\": {");
            sb.append("\"").append(field).append("\":").append("\"").append(term).append("\"");
            sb.append("}\n" +
                    "            }\n" +
                    "          }\n" +
                    "        }");
            if (i!=terms.size()-1){
                sb.append(",");
            }

        }
        sb.append("],\"minimum_should_match\":");
        sb.append("\"").append(percentage).append("%").append("\"").append("}}").append(",");


        sb.append("\"filter\": {\n" +
                "        \"ids\": {\n" +
                "          \"values\": [");
        for (int i=0;i<ids.length;i++){
            sb.append("\"").append(ids[i]).append("\"");
            if (i!=ids.length-1){
                sb.append(",");
            }
        }
        sb.append("]}}}}");


        return submitQuery(sb.toString());

    }

    /**
     * use as an inverted index
     * no score is computed
     * @param term stemmed term
     * @return
     * @throws Exception
     */
    public List<String> termFilter(String field, String term, String[] ids) throws Exception{
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }

        /**
         * setSize() has a huge impact on performance, the smaller the faster
         */

        TermFilterBuilder termFilterBuilder = new TermFilterBuilder(field, term);
        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);


        idsFilterBuilder.addIds(ids);


        SearchResponse response = client.prepareSearch(indexName).setSize(ids.length).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.constantScoreQuery(
                        FilterBuilders.andFilter(termFilterBuilder,
                                idsFilterBuilder))).
                execute().actionGet();
        List<String> list = new ArrayList<>(response.getHits().getHits().length);
        for (SearchHit searchHit : response.getHits()) {
            list.add(searchHit.getId());
        }
        if(logger.isDebugEnabled()){
            logger.debug("time spent on termFilter() for " + term + " = " + stopWatch+
                    " There are "+list.size()+" matched docs");
        }
        return list;
    }





    public Set<String> listAllFields() throws Exception{
        GetMappingsResponse response = client.admin().indices().prepareGetMappings(this.indexName).
                execute().actionGet();
        MappingMetaData mappingMetaData = response.getMappings().get(this.indexName).get(this.documentType);
        Map map = (Map)mappingMetaData.getSourceAsMap().get("properties");
        Set<String> fields = new HashSet<>();
        for (Object field: map.keySet()){
            fields.add(field.toString());
        }
        return fields;
    }

    public String getFieldType(String field) {
        GetFieldMappingsResponse response = this.client.admin().indices().
                prepareGetFieldMappings(this.indexName).setTypes(this.documentType).
                setFields(field).execute().actionGet();
        Map map = (Map)response.mappings().get(this.indexName).get(this.documentType).
                get(field).sourceAsMap().get(field);
        return map.get("type").toString();
    }

    public List<String> getStringListField(String id, String field){
        Object object = getField(id,field);
        if (object==null){
            return new ArrayList<>();
        }
        return getListField(id,field).stream().map(obj -> obj.toString())
                .collect(Collectors.toList());
    }

    // should not support
//    public List<Integer> getIntListField(String id, String field){
//        return getListField(id,field).stream().map(object -> Integer.parseInt(object.toString()))
//                .collect(Collectors.toList());
//    }

    public List<Float> getFloatListField(String id, String field){
        return getListField(id,field).stream().map(object -> Float.parseFloat(object.toString()))
                .collect(Collectors.toList());
    }

    public String getStringField(String id, String field){
        Object object = getField(id,field);
        if (object==null){
            return STRING_MISSING_VALUE;
        }
        return object.toString();
    }

    // it seems we should allow int field, as it seems difficult to handle missing values
    public int getIntField(String id, String field){
        return Integer.parseInt(getField(id,field).toString());
    }

    public float getFloatField(String id, String field){
        Object object = getField(id,field);
        if (object==null){
            return Float.NaN;
        }
        return Float.parseFloat(object.toString());
    }

    /**
     * return all documents within ids that miss the field
     * @param field
     * @param ids
     * @return
     */
    public List<String> docsWithFieldMissing(String field, String[] ids){
        List<String> docs = new ArrayList<>();
        for (String id: ids){
            Object object = getField(id,field);
            if (object==null){
                docs.add(id);
            }
        }
        return docs;
    }

    /**
     * df is from one shard!!!
     * @param id
     * @return term statistics from one doc
     * @throws IOException
     */
    public Set<TermStat> getTermStats(String id) throws IOException {
        return getTermStats(this.bodyField,id);
    }


    /**
     * df is from one shard!!!
     * @param id
     * @return term statistics from one doc
     * @throws IOException
     */
    public Set<TermStat> getTermStats(String field, String id) throws IOException {
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        TermVectorResponse response = client.prepareTermVector(indexName, documentType, id)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setTermStatistics(true)
                .setSelectedFields(field).
                        execute().actionGet();

        Terms terms = response.getFields().terms(field);
        Set<TermStat> set = new HashSet<>();
        // if the field is empty, terms==null
        if (terms==null){
            return set;
        }
        TermsEnum iterator = terms.iterator(null);


        for (int i=0;i<terms.size();i++){
            String term = iterator.next().utf8ToString();
            int tf = iterator.docsAndPositions(null,null).freq();
            int df = iterator.docFreq();
            DefaultSimilarity defaultSimilarity = new DefaultSimilarity();
            /**
             * from lucene
             */
            /**
             * tf is just tf, not square root of tf as in lucene
             */
            /** Implemented as <code>log(numDocs/(docFreq+1)) + 1</code>. */
            float tfidf = tf*defaultSimilarity.idf(df,this.numDocs);
            TermStat termStat = new TermStat(term);
            termStat.setTf(tf).setDf(df).setTfidf(tfidf);
            set.add(termStat);

        }

        if(logger.isDebugEnabled()){
            logger.debug("time spent on getNgramInfos for "+id+" = " + stopWatch);
        }
        return set;
    }

    /**
     * naive implementation
     * @param id
     * @return
     */
    public int getDocLength(String id){
        return getTermVector(id).keySet().size();
    }


    public Map<Integer,String> getTermVector(String id){
        Map<Integer,String> termVector = null;
        try {
            termVector = this.termVectorCache.get(id);
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return termVector;
    }

    public Map<Integer,String> getTermVectorFromIndex(String field, String id){
        Map<Integer,String> map = null;
        try {
            map = getTermVectorWithException(field, id);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return map;
    }

    public Map<Integer,String> getTermVectorFromIndex(String id){
        return getTermVectorFromIndex(this.bodyField,id);
    }

    private Map<Integer,String> getTermVectorWithException(String field, String id) throws IOException {
        TermVectorResponse response = client.prepareTermVector(indexName, documentType, id)
                .setOffsets(false).setPositions(true).setFieldStatistics(false)
                .setTermStatistics(false)
                .setSelectedFields(field).
                        execute().actionGet();

        Map<Integer,String> map = new HashMap<>();
        Terms terms = response.getFields().terms(field);
        if (terms==null){
            return map;
        }
        TermsEnum iterator = terms.iterator(null);
        for (int i=0;i<terms.size();i++){
            String term = iterator.next().utf8ToString();
            int tf = iterator.docsAndPositions(null, null).freq();
            //must declare docsAndPositionsEnum as a local variable and reuse it for positions
            DocsAndPositionsEnum docsAndPositionsEnum = iterator.docsAndPositions(null, null);
            for (int j=0;j<tf;j++){
                int pos = docsAndPositionsEnum.nextPosition();
                map.put(pos,term);
            }
        }
        return map;
    }




    public void close() {
        this.client.close();
        if (this.clientType.equals("node")){
            this.node.close();
        }
        //todo
        System.out.println("closed");
    }




    protected int fetchNumDocs() throws Exception{
        //todo cast saft?
        //todo  this admin method seems buggy!
//        return (int)client.admin().indices().prepareStats(this.indexName)
//                .get().getIndex(this.indexName).getTotal().termFilter().getCount();
        return (int)client.prepareCount(indexName).setQuery(QueryBuilders.matchAllQuery()).execute()
                .actionGet().getCount();
//        SearchResponse response = client.prepareSearch(indexName).setSize(Integer.MAX_VALUE).
//                addField("").
//                setQuery(QueryBuilders.matchAllQuery()).execute().actionGet();
//        //todo safe?
//        return (int)response.getHits().totalHits();
    }

    public boolean hasField(String id, String field){
        return getField(id, field) != null;
    }

    public Object getField(String id, String field){
        GetResponse response = client.prepareGet(this.indexName, this.documentType, id).
                setFields(field)
                .execute()
                .actionGet();
        if (response==null){
            if (logger.isWarnEnabled()){
                logger.warn("no response from document "+id+" when fetching field "+field+"!");
            }
            return null;
        }else if (response.getField(field)==null){
            if (logger.isWarnEnabled()) {
                logger.warn("document " + id + " has no field " + field + "!");
            }
            return null;
        }

        return response.getField(field).getValue();
    }

    //todo handle missing values
    public List<Object> getListField(String id, String field){
        GetResponse response = client.prepareGet(this.indexName, this.documentType, id).
                setFields(field)
                .execute()
                .actionGet();
        if (response==null){
            if (logger.isWarnEnabled()){
                logger.warn("no response from document "+id+" when fetching field "+field+"!");
            }
            return new ArrayList<>();
        }else if (response.getField(field)==null){
            if (logger.isWarnEnabled()) {
                logger.warn("document " + id + " has no field " + field + "!");
            }
            return new ArrayList<>();
        }
        return response.getField(field).getValues();
    }


    /**
     * phrase query
     * use whitespace analyzer in the query
     * @param field
     * @param phrase already stemmed
     * @param slop
     * @return
     */
    public SearchResponse matchPhrase(String field, String phrase, int slop){

        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.matchPhraseQuery(field, phrase).slop(slop)
                    .analyzer("whitespace")).
                execute().actionGet();
        return response;

//        debug
//        XContentBuilder builder = XContentFactory.jsonBuilder();
//        builder.startObject();
//        System.out.println(response.toXContent(builder, ToXContent.EMPTY_PARAMS));
//        builder.endObject();
//        System.out.println(builder.string());
    }


    /**
     * phrase query
     * use whitespace analyzer in the query
     * @param field
     * @param phrase already stemmed
     * @param ids
     * @param slop
     * @return
     */
    public SearchResponse matchPhrase(String field, String phrase,
                                      String[] ids, int slop) {
        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);
        idsFilterBuilder.addIds(ids);

        SearchResponse response = client.prepareSearch(indexName).setSize(ids.length).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchPhraseQuery(field, phrase)
                                .slop(slop).analyzer("whitespace"),
                        idsFilterBuilder))
                .execute().actionGet();


//        debug
//        XContentBuilder builder = XContentFactory.jsonBuilder();
//        builder.startObject();
//        System.out.println(response.toXContent(builder, ToXContent.EMPTY_PARAMS));
//        builder.endObject();
//        System.out.println(builder.string());

        return response;
    }

    public SearchResponse spanNear(Ngram ngram){
        String field = ngram.getField();
        int slop = ngram.getSlop();
        boolean inOrder = ngram.isInOrder();
        SpanNearQueryBuilder queryBuilder = QueryBuilders.spanNearQuery();
        for (String term: ngram.getTerms()){
            queryBuilder.clause(new SpanTermQueryBuilder(field, term));
        }
        queryBuilder.inOrder(inOrder);
        queryBuilder.slop(slop);

        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(queryBuilder)
                .execute().actionGet();

        return response;

    }

    public Collection<org.elasticsearch.search.aggregations.bucket.terms.Terms.Bucket> termAggregation(String field){
        SearchResponse response = client.prepareSearch(indexName)
                //no return for matches
                .setSize(0)
                .setQuery(QueryBuilders.matchAllQuery())
                //return all terms
                .addAggregation(terms("agg").field(field).size(Integer.MAX_VALUE))
                .execute().actionGet();
        org.elasticsearch.search.aggregations.bucket.terms.Terms terms = response.getAggregations().get("agg");
        Collection<org.elasticsearch.search.aggregations.bucket.terms.Terms.Bucket> buckets = terms.getBuckets();
        return buckets;
    }


    public Collection<org.elasticsearch.search.aggregations.bucket.terms.Terms.Bucket> termAggregation(String field, String[] ids){
        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);
        idsFilterBuilder.addIds(ids);

        SearchResponse response = client.prepareSearch(indexName)
                //no return for matches
                .setSize(0)
                .setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchAllQuery(),
                        idsFilterBuilder))
                        //return all terms
                .addAggregation(terms("agg").field(field).size(Integer.MAX_VALUE))
                .execute().actionGet();
        org.elasticsearch.search.aggregations.bucket.terms.Terms terms = response.getAggregations().get("agg");
        Collection<org.elasticsearch.search.aggregations.bucket.terms.Terms.Bucket> buckets = terms.getBuckets();
        return buckets;
    }

    public long count(Ngram ngram){
        String field = ngram.getField();
        int slop = ngram.getSlop();
        boolean inOrder = ngram.isInOrder();
        SpanNearQueryBuilder queryBuilder = QueryBuilders.spanNearQuery();
        for (String term: ngram.getTerms()){
            queryBuilder.clause(new SpanTermQueryBuilder(field, term));
        }
        queryBuilder.inOrder(inOrder);
        queryBuilder.slop(slop);

        long hits = client.prepareCount(this.indexName).setQuery(queryBuilder)
                .execute().actionGet().getCount();
        return hits;
    }

    public long count(Ngram ngram, String[] ids){
        String field = ngram.getField();
        int slop = ngram.getSlop();
        boolean inOrder = ngram.isInOrder();
        SpanNearQueryBuilder queryBuilder = QueryBuilders.spanNearQuery();
        for (String term: ngram.getTerms()){
            queryBuilder.clause(new SpanTermQueryBuilder(field, term));
        }
        queryBuilder.inOrder(inOrder);
        queryBuilder.slop(slop);

        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);
        idsFilterBuilder.addIds(ids);

        long hits = client.prepareCount(this.indexName).setQuery(queryBuilder)
                .execute().actionGet().getCount();
        return hits;
    }

    public SearchResponse spanNear(Ngram ngram, int size){
        String field = ngram.getField();
        int slop = ngram.getSlop();
        boolean inOrder = ngram.isInOrder();
        SpanNearQueryBuilder queryBuilder = QueryBuilders.spanNearQuery();
        for (String term: ngram.getTerms()){
            queryBuilder.clause(new SpanTermQueryBuilder(field, term));
        }
        queryBuilder.inOrder(inOrder);
        queryBuilder.slop(slop);

        SearchResponse response = client.prepareSearch(indexName).setSize(size).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(queryBuilder)
                .execute().actionGet();
        System.out.println(response.getHits().getTotalHits());

        return response;

    }


    public SearchResponse spanNear(Ngram ngram, String[] ids){
        String field = ngram.getField();
        int slop = ngram.getSlop();
        boolean inOrder = ngram.isInOrder();
        SpanNearQueryBuilder queryBuilder = QueryBuilders.spanNearQuery();
        for (String term: ngram.getTerms()){
            queryBuilder.clause(new SpanTermQueryBuilder(field, term));
        }
        queryBuilder.inOrder(inOrder);
        queryBuilder.slop(slop);

        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);
        idsFilterBuilder.addIds(ids);

        SearchResponse response = client.prepareSearch(indexName).setSize(ids.length).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(queryBuilder, idsFilterBuilder))
                        .execute().actionGet();

        return response;

    }

    public SearchResponse spanNearFrequency(Ngram ngram, String[] ids){
        String field = ngram.getField();
        int slop = ngram.getSlop();
        boolean inOrder = ngram.isInOrder();
        SpanNearQueryBuilder queryBuilder = QueryBuilders.spanNearQuery();
        for (String term: ngram.getTerms()){
            queryBuilder.clause(new SpanTermQueryBuilder(field, term));
        }
        queryBuilder.inOrder(inOrder);
        queryBuilder.slop(slop);

        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);
        idsFilterBuilder.addIds(ids);


        //todo: hanle ngram frequency properly
        Map<String,Object> params = new HashMap<>();
        params.put("field",ngram.getField());
        params.put("term",ngram.getTerms()[0]);

        SearchResponse response = client.prepareSearch(indexName).setSize(ids.length).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.functionScoreQuery(QueryBuilders.filteredQuery(queryBuilder, idsFilterBuilder),
                        ScoreFunctionBuilders.scriptFunction("getTF","groovy",params))
                        .boostMode(CombineFunction.REPLACE))
                        .execute().actionGet();

        return response;

    }


    public SearchResponse spanNot(SpanNotNgram ngram, String[] ids){
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
        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);
        idsFilterBuilder.addIds(ids);

        SearchResponse response = client.prepareSearch(indexName).setSize(ids.length).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(spanNotQueryBuilder, idsFilterBuilder))
                .execute().actionGet();

        return response;

    }

    public SearchResponse spanNot(SpanNotNgram ngram){
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
        //todo: upgrade to 1.5
//                .pre(pre).post(post);


        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(spanNotQueryBuilder)
                .execute().actionGet();

        return response;

    }


    /**
     * simple match
     * use whitespace analyzer
     * @param field
     * @param phrase already stemmed
     * @param operator and /or
     * @return
     */
    public SearchResponse match(String field, String phrase, MatchQueryBuilder.Operator operator){

        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.matchQuery(field, phrase).operator(operator)
                        .analyzer("whitespace")).
                execute().actionGet();
        return response;

//        debug
//        XContentBuilder builder = XContentFactory.jsonBuilder();
//        builder.startObject();
//        System.out.println(response.toXContent(builder, ToXContent.EMPTY_PARAMS));
//        builder.endObject();
//        System.out.println(builder.string());
    }

    /**
     * simple match
     * use whitespace analyzer
     * @param field
     * @param phrase already stemmed
     * @param ids
     * @param operator
     * @return
     */
    public SearchResponse match(String field, String phrase, String[] ids,
                                MatchQueryBuilder.Operator operator){
        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);
        idsFilterBuilder.addIds(ids);
        SearchResponse response = client.prepareSearch(indexName).setSize(ids.length).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchQuery(field, phrase)
                                .operator(operator).analyzer("whitespace"),
                        idsFilterBuilder)).
                execute().actionGet();
        return response;
    }





    public long phraseDF(String field, String phrase, int slop){
        SearchResponse response = this.matchPhrase(field,phrase,slop);
        return response.getHits().getTotalHits();

    }



    public long DF(String field, String phrase, MatchQueryBuilder.Operator operator){
        SearchResponse response = this.match(field,phrase,operator);
        return response.getHits().getTotalHits();

    }

    /**
     * analyze the given text using the provided analyzer, return an ngram
     * @param text
     * @param analyzer
     * @return
     */
    public Ngram analyze(String text, String analyzer){
        List<AnalyzeResponse.AnalyzeToken> tokens = client.admin().indices().prepareAnalyze(indexName,text).setAnalyzer(analyzer).get().getTokens();

        Ngram ngram = new Ngram();
        StringBuilder sb = new StringBuilder();
        for (int i=0;i<tokens.size();i++)
        {
            AnalyzeResponse.AnalyzeToken token = tokens.get(i);
            sb.append(token.getTerm());
            if (i!=tokens.size()-1){
                sb.append(" ");
            }
        }
        ngram.setNgram(sb.toString());
        return ngram;

    }

    /**
     * analyze the given text using the provided analyzer, return an ngram
     * @param text
     * @param analyzer
     * @return
     */
    public List<String> analyzeString(String text, String analyzer){
        List<AnalyzeResponse.AnalyzeToken> tokens = client.admin().indices().prepareAnalyze(indexName,text).setAnalyzer(analyzer).get().getTokens();
        List<String> list = new ArrayList<>();
        for (int i=0;i<tokens.size();i++)
        {
            AnalyzeResponse.AnalyzeToken token = tokens.get(i);
            list.add(token.getTerm());

        }
        return list;

    }





    //=================old implementations========================

//    /**
//     * n is ignored for now, always return unigrams
//     *
//     * @param n
//     * @param field
//     * @return
//     */
//    public Set<Ngram> getNgrams(int n, String field) {
//        SearchResponse response = client.prepareSearch(this.indexName)
//                .addAggregation(AggregationBuilders.terms("ngrams").field(field).size(1000000000))
//                .execute().actionGet();
//        Terms terms = response.getAggregations().get("ngrams");
//        Collection<Terms.Bucket> buckets = terms.getBuckets();
//
//        Set<Ngram> set = new HashSet<Ngram>();
//        for (Terms.Bucket bucket : buckets) {
//            set.add(new Ngram(bucket.getKey()));
//        }
//        return set;
//    }

//
//    /**
//     * n is ignored for now, always return unigrams
//     *
//     * @param n
//     * @return
//     */
//    public Set<Ngram> getNgrams(int n) {
//        return getNgrams(n, this.bodyField);
//    }
//    /**
//     * old implementation based on jason response
//     * @param n =1 for now
//     * @return
//     */
//    public Set<Ngram> getNgrams(String id, int n) {
//        StopWatch stopWatch=null;
//        if(logger.isDebugEnabled()){
//            stopWatch = new StopWatch();
//            stopWatch.start();
//        }
//        String termVectorResponse = null;
//        try {
//            termVectorResponse = getTermVectorResponse(documentType,this.bodyField,id);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//        TermVectorResponsePOJO termVectorResponsePOJO = null;
//        try {
//            termVectorResponsePOJO = parseTermVectorResponse(termVectorResponse);
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//        Set<String> terms = termVectorResponsePOJO.getAllTerms();
//        Set<Ngram> ngramSet = new HashSet<>(terms.size());
//        for (String term: terms){
//            ngramSet.add(new Ngram(term));
//        }
//        if(logger.isDebugEnabled()){
//            logger.debug("time spent on getNgrams from doc "+id+" = "+stopWatch+
//                    " It has "+ngramSet.size()+" ngrams");
//        }
//        return ngramSet;
//    }



//    public Set<Integer> termFilter(Ngram ngram) throws Exception{
//
//        StopWatch stopWatch = new StopWatch();
//        stopWatch.start();
//        String termFilterResponse = termFilter(ngram.getFeatureName());
//
//        TermFilterResponsePOJO termFilterResponsePOJO = parseTermFilterResponse(termFilterResponse);
//        System.out.println("time spent on termFilter() for "+ngram.getFeatureName()+stopWatch);
//        return termFilterResponsePOJO.getAllIds();
//
//    }



//    /**
//     * experimental
//     * @param ngram
//     * @return
//     * @throws Exception
//     */
//    //todo using a set seems to be slow, maybe fine
//    public Set<Integer> termFilter(Ngram ngram) throws Exception{
//        StopWatch stopWatch=null;
//        if(logger.isDebugEnabled()){
//            stopWatch = new StopWatch();
//            stopWatch.start();
//        }
//        Set<Integer> set = new HashSet<>();
//        /**
//         * setSize() has a huge impact on performance, the smaller the faster
//         */
//        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).setHighlighterFilter(false).setTrackScores(false).
//                setNoFields().setExplain(false).setFetchSource(false).
//                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchAllQuery(),FilterBuilders.termFilter(this.bodyField,ngram.getFeatureName()).cache(true))).
//                execute().actionGet();
//        for (SearchHit searchHit : response.getHits()) {
//            set.add(Integer.parseInt(searchHit.getId()));
//        }
//        if(logger.isDebugEnabled()){
//            logger.debug("time spent on termFilter() for " + ngram.getFeatureName() + " = " + stopWatch+
//                    " There are "+set.size()+" matched docs");
//        }
//        return set;
//    }


//    public List<String> termFilter(Ngram ngram, String[] ids) throws Exception{
//        StopWatch stopWatch=null;
//        if(logger.isDebugEnabled()){
//            stopWatch = new StopWatch();
//            stopWatch.start();
//        }
//
//        /**
//         * setSize() has a huge impact on performance, the smaller the faster
//         */
//
//        //todo reuse idsFilterBuilder
//        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder(documentType);
//
//
//        idsFilterBuilder.addIds(ids);
//
//        TermFilterBuilder termFilterBuilder = new TermFilterBuilder(this.bodyField, ngram.getFeatureName());
//
//        SearchResponse response = client.prepareSearch(indexName).setSize(ids.length).
//                setHighlighterFilter(false).setTrackScores(false).
//                setNoFields().setExplain(false).setFetchSource(false).
//                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchAllQuery(),
//                        FilterBuilders.andFilter(termFilterBuilder,
//                                idsFilterBuilder))).
//                execute().actionGet();
//        List<String> list = new ArrayList<>(response.getHits().getHits().length);
//        for (SearchHit searchHit : response.getHits()) {
//            list.add(searchHit.getId());
//        }
//        if(logger.isDebugEnabled()){
//            logger.debug("time spent on termFilter() for " + ngram.getFeatureName() + " = " + stopWatch+
//                    " There are "+list.size()+" matched docs");
//        }
//        return list;
//    }

//    public List<String> termFilter(Ngram ngram, String[] ids, IdsFilterBuilder idsFilterBuilder) throws Exception{
//        StopWatch stopWatch=null;
//        if(logger.isDebugEnabled()){
//            stopWatch = new StopWatch();
//            stopWatch.start();
//        }
//
//        /**
//         * setSize() has a huge impact on performance, the smaller the faster
//         */
//
//        //todo reuse idsFilterBuilder
//
//
//        TermFilterBuilder termFilterBuilder = new TermFilterBuilder(this.bodyField, ngram.getFeatureName());
//
//        SearchResponse response = client.prepareSearch(indexName).setSize(ids.length).
//                setHighlighterFilter(false).setTrackScores(false).
//                setNoFields().setExplain(false).setFetchSource(false).
//                setQuery(QueryBuilders.constantScoreQuery(
//                        FilterBuilders.andFilter(termFilterBuilder,
//                                idsFilterBuilder))).
//                execute().actionGet();
//        List<String> list = new ArrayList<>(response.getHits().getHits().length);
//        for (SearchHit searchHit : response.getHits()) {
//            list.add(searchHit.getId());
//        }
//        if(logger.isDebugEnabled()){
//            logger.debug("time spent on termFilter() for " + ngram.getFeatureName() + " = " + stopWatch+
//                    " There are "+list.size()+" matched docs");
//        }
//        return list;
//    }



    public static class Builder {
        private String indexName = "unknown_index";
        private String documentType = "document";
        private String clientType = "transport";
        private String clusterName = "elasticsearch";
        private String bodyField = "body";
        private List<String> hosts = new ArrayList<>();
        private List<Integer> ports = new ArrayList<>();
        private int termVectorCacheSize = 10000;



        public Builder setIndexName(String indexName) {
            this.indexName = indexName;
            return this;
        }

        public Builder setDocumentType(String documentType) {
            this.documentType = documentType;
            return this;
        }

        public Builder setClientType(String clientType) {
            this.clientType = clientType;
            return this;
        }

        public Builder setClusterName(String clusterName) {
            this.clusterName = clusterName;
            return this;
        }

        public Builder addHostAndPort(String host, int port){
            this.hosts.add(host);
            this.ports.add(port);
            return this;
        }

        public Builder addHostsAndPorts(String[] hosts, String[] ports){
            for (int i=0;i< hosts.length;i++){
                addHostAndPort(hosts[i],Integer.parseInt(ports[i]));
            }
            return this;
        }

        public Builder setBodyField(String bodyField) {
            this.bodyField = bodyField;
            return this;
        }

        public Builder setTermVectorCacheSize(int termVectorCacheSize) {
            this.termVectorCacheSize = termVectorCacheSize;
            return this;
        }


        public ESIndex build() throws Exception {
            boolean legal = (clientType.equals("node"))||(clientType.equals("transport"));
            if (!legal){
                throw new IllegalArgumentException("clientType = node or transport");
            }
            ESIndex esIndex = new ESIndex();
            esIndex.indexName = indexName;
            esIndex.documentType = documentType;
            esIndex.clientType = clientType;
            esIndex.clusterName = clusterName;
            esIndex.bodyField = bodyField;


            if (clientType.equals("node")){
                /**
                 * don't hold data
                 */
                Node node = nodeBuilder().client(true).
                        clusterName(clusterName).node();
                esIndex.node = node;
                esIndex.client = node.client();
            } else {
                Settings settings = ImmutableSettings.settingsBuilder()
                        .put("cluster.name", clusterName).build();

                esIndex.client = new TransportClient(settings);
                for (int i=0;i<this.hosts.size();i++){
                    ((TransportClient)esIndex.client)
                            .addTransportAddress(new InetSocketTransportAddress(this.hosts.get(i),
                                    this.ports.get(i)));
                }
            }
            esIndex.numDocs = esIndex.fetchNumDocs();

            esIndex.termVectorCache = CacheBuilder.newBuilder()
                    .maximumSize(this.termVectorCacheSize)
                    .build(new CacheLoader<String, Map<Integer, String>>() {
                        @Override
                        public Map<Integer, String> load(String id) throws Exception {
                            return esIndex.getTermVectorFromIndex(id);
                        }
                    });

            return esIndex;
        }

    }

}
