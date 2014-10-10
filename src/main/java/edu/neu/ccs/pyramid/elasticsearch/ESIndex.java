package edu.neu.ccs.pyramid.elasticsearch;

import com.google.common.cache.LoadingCache;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.DocsAndPositionsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.similarities.DefaultSimilarity;
import org.elasticsearch.action.admin.indices.mapping.get.GetFieldMappingsResponse;
import org.elasticsearch.action.admin.indices.mapping.get.GetMappingsResponse;
import org.elasticsearch.action.get.GetResponse;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.termvector.TermVectorResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.cluster.metadata.MappingMetaData;
import org.elasticsearch.index.query.*;
import org.elasticsearch.node.Node;
import org.elasticsearch.search.SearchHit;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;


/**
 * Created by chengli on 8/20/14.
 */
public class ESIndex {
    private static final Logger logger = LogManager.getLogger();

    Client client;
    Node node;
    String indexName;
    int numDocs;
    String labelField;
    String extLabelField;
    String documentType;
    String clientType;
    String clusterName;
    String bodyField;
    /**
     * concurrent LRU cache for termvectors
     */
    LoadingCache<String,Map<Integer,String>> termVectorCache;


    public int getNumDocs() {
        return numDocs;
    }

    public Client getClient() {
        return client;
    }

    public String getIndexName() {
        return indexName;
    }

    public String getLabelField() {
        return labelField;
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
            logger.debug("time spent on getDocs() for " + term+ " = " + stopWatch+
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
    public List<String> getDocs(String term) throws Exception{
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }

        /**
         * setSize() has a huge impact on performance, the smaller the faster
         */

        TermFilterBuilder termFilterBuilder = new TermFilterBuilder(this.bodyField, term);

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
            logger.debug("time spent on getDocs() for " + term + " = " + stopWatch+
                    " There are "+list.size()+" matched docs");
        }
        return list;
    }


    public int getLabel(String id){
        GetResponse response = client.prepareGet(indexName, documentType, id).setFields(this.labelField)
                .execute()
                .actionGet();
        if (logger.isDebugEnabled()){
            logger.debug("getting label from id "+id+", field "+this.labelField);
        }
        return (int) response.getField(this.labelField).getValue();
    }

    public String getExtLabel(String id){
        GetResponse response = client.prepareGet(indexName, documentType, id).setFields(this.extLabelField)
                .execute()
                .actionGet();
        return (String) response.getField(this.extLabelField).getValue();
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
        return getListField(id,field).stream().map(object -> (String) object)
                .collect(Collectors.toList());
    }

    public List<Integer> getIntListField(String id, String field){
        return getListField(id,field).stream().map(object -> Integer.parseInt(object.toString()))
                .collect(Collectors.toList());
    }

    public List<Float> getFloatListField(String id, String field){
        return getListField(id,field).stream().map(object -> Float.parseFloat(object.toString()))
                .collect(Collectors.toList());
    }

    public String getStringField(String id, String field){
        return getField(id,field).toString();
    }

    public int getIntField(String id, String field){
        return Integer.parseInt(getField(id,field).toString());
    }

    public float getFloatField(String id, String field){
        return Float.parseFloat(getField(id,field).toString());
    }

    /**
     * df is from one shard!!!
     * @param id
     * @return term statistics from one doc
     * @throws IOException
     */
    public Set<TermStat> getTermStats(String id) throws IOException {
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        TermVectorResponse response = client.prepareTermVector(indexName, documentType, id)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setTermStatistics(true)
                .setSelectedFields(this.bodyField).
                        execute().actionGet();

        Terms terms = response.getFields().terms(this.bodyField);
        TermsEnum iterator = terms.iterator(null);

        Set<TermStat> set = new HashSet<>();
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

    public Map<Integer,String> getTermVector(String id){
        Map<Integer,String> termVector = null;
        try {
            termVector = this.termVectorCache.get(id);
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
        return termVector;
    }

    Map<Integer,String> getTermVectorFromIndex(String id){
        Map<Integer,String> map = null;
        try {
            map = getTermVectorWithException(id);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return map;
    }

    private Map<Integer,String> getTermVectorWithException(String id) throws IOException {
        TermVectorResponse response = client.prepareTermVector(indexName, documentType, id)
                .setOffsets(false).setPositions(true).setFieldStatistics(false)
                .setTermStatistics(false)
                .setSelectedFields(this.bodyField).
                        execute().actionGet();

        Map<Integer,String> map = new HashMap<>();
        Terms terms = response.getFields().terms(this.bodyField);
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
    }




    protected int fetchNumDocs() throws Exception{
        //todo cast saft?
        //todo  this admin method seems buggy!
//        return (int)client.admin().indices().prepareStats(this.indexName)
//                .get().getIndex(this.indexName).getTotal().getDocs().getCount();
        SearchResponse response = client.prepareSearch(indexName).setSize(10000000).
                addField("").
                setQuery(QueryBuilders.matchAllQuery()).execute().actionGet();
        //todo safe?
        return (int)response.getHits().totalHits();
    }

    public Object getField(String id, String field){
        GetResponse response = client.prepareGet(this.indexName, this.documentType, id).
                setFields(field)
                .execute()
                .actionGet();
        if (logger.isErrorEnabled()){
            if (response==null){
                logger.error("no response from document "+id+" when fetching field "+field+"!");
            }
            else if (response.getField(field)==null){
                logger.error("document "+id+" has no field "+field+"!");
            }
        }
        return response.getField(field).getValue();
    }

    public List<Object> getListField(String id, String field){
        GetResponse response = client.prepareGet(this.indexName, this.documentType, id).
                setFields(field)
                .execute()
                .actionGet();
        if (logger.isErrorEnabled()){
            if (response==null){
                logger.error("no response from document "+id+" when fetching field "+field+"!");
            }
            else if (response.getField(field)==null){
                logger.error("document "+id+" has no field "+field+"!");
            }
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

        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
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
        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchQuery(field, phrase)
                                .operator(operator).analyzer("whitespace"),
                        idsFilterBuilder)).
                execute().actionGet();
        return response;
    }

    /**
     * use whitespace analyzer
     * @param bodyField
     * @param phrase already stemmed
     * @param slop
     * @param labelField
     * @param label
     * @return
     */
    public SearchResponse matchPhraseForClass(String bodyField, String phrase,
                                              int slop,
                                              String labelField, int label){
        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchPhraseQuery(bodyField,phrase)
                                .slop(slop).analyzer("whitespace"),
                        FilterBuilders.termFilter(labelField,label))).
                execute().actionGet();

        //        debug
//        XContentBuilder builder = XContentFactory.jsonBuilder();
//        builder.startObject();
//        System.out.println(response.toXContent(builder, ToXContent.EMPTY_PARAMS));
//        builder.endObject();
//        System.out.println(builder.string());

        return response;
    }

    public SearchResponse matchForClass(String bodyField, String phrase,
                                        MatchQueryBuilder.Operator operator,
                                        String labelField, int label){
        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(false).setFetchSource(false).
                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchQuery(bodyField,phrase)
                                .operator(operator).analyzer("whitespace"),
                        FilterBuilders.termFilter(labelField,label))).
                execute().actionGet();

        //        debug
//        XContentBuilder builder = XContentFactory.jsonBuilder();
//        builder.startObject();
//        System.out.println(response.toXContent(builder, ToXContent.EMPTY_PARAMS));
//        builder.endObject();
//        System.out.println(builder.string());

        return response;
    }

    public long phraseDF(String field, String phrase, int slop){
        SearchResponse response = this.matchPhrase(field,phrase,slop);
        return response.getHits().getTotalHits();

    }

    public long phraseDFForClass(String bodyField, String phrase,
                                 int slop,
                                 String labelField, int label) {
        SearchResponse response = this.matchPhraseForClass(bodyField,phrase,
                slop,labelField,label);
        return response.getHits().getTotalHits();
    }

    public long DF(String field, String phrase, MatchQueryBuilder.Operator operator){
        SearchResponse response = this.match(field,phrase,operator);
        return response.getHits().getTotalHits();

    }

    public long DFForClass(String bodyField, String phrase,
                           MatchQueryBuilder.Operator operator,
                           String labelField, int label) {
        SearchResponse response = this.matchForClass(bodyField,phrase,operator,labelField,label);
        return response.getHits().getTotalHits();
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



//    public Set<Integer> getDocs(Ngram ngram) throws Exception{
//
//        StopWatch stopWatch = new StopWatch();
//        stopWatch.start();
//        String termFilterResponse = termFilter(ngram.getFeatureName());
//
//        TermFilterResponsePOJO termFilterResponsePOJO = parseTermFilterResponse(termFilterResponse);
//        System.out.println("time spent on getDocs() for "+ngram.getFeatureName()+stopWatch);
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
//    public Set<Integer> getDocs(Ngram ngram) throws Exception{
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
//            logger.debug("time spent on getDocs() for " + ngram.getFeatureName() + " = " + stopWatch+
//                    " There are "+set.size()+" matched docs");
//        }
//        return set;
//    }


//    public List<String> getDocs(Ngram ngram, String[] ids) throws Exception{
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
//            logger.debug("time spent on getDocs() for " + ngram.getFeatureName() + " = " + stopWatch+
//                    " There are "+list.size()+" matched docs");
//        }
//        return list;
//    }

//    public List<String> getDocs(Ngram ngram, String[] ids, IdsFilterBuilder idsFilterBuilder) throws Exception{
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
//            logger.debug("time spent on getDocs() for " + ngram.getFeatureName() + " = " + stopWatch+
//                    " There are "+list.size()+" matched docs");
//        }
//        return list;
//    }

}
