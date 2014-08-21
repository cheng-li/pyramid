package edu.neu.ccs.pyramid.elasticsearch;

import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.feature.NgramInfo;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.similarities.DefaultSimilarity;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.action.admin.indices.mapping.get.GetFieldMappingsResponse;
import org.elasticsearch.action.admin.indices.mapping.get.GetMappingsResponse;
import org.elasticsearch.action.get.GetResponse;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.action.termvector.TermVectorResponse;
import org.elasticsearch.client.Client;
import org.elasticsearch.cluster.metadata.MappingMetaData;
import org.elasticsearch.common.xcontent.ToXContent;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentFactory;
import org.elasticsearch.index.query.*;
import org.elasticsearch.node.Node;
import org.elasticsearch.search.SearchHit;

import java.io.IOException;
import java.util.*;

import static org.elasticsearch.node.NodeBuilder.nodeBuilder;

/**
 * Created by chengli on 8/20/14.
 */
public class ESIndex {
    private static final Logger logger = LogManager.getLogger();

    private Client client;
    private Node node;
    private String indexName;
    private int numDocs;
    private String labelField = "label";
    private String extLabelField = "real_label";
    private String type = "document";


    public ESIndex(String indexName,String clusterName) throws Exception{
//        Settings settings = ImmutableSettings.settingsBuilder()
//                .put("http.enabled", "false")
//                .put("transport.tcp.port", "9300-9400")
//                .put("discovery.zen.ping.multicast.enabled", "false")
//                .put("discovery.zen.ping.unicast.hosts", "fiji11:9300").build();

        this.indexName = indexName;
        /**
         * don't hold data
         */
        Node node = nodeBuilder().client(true).
//                settings(settings).
        clusterName(clusterName).node();
        this.node = node;
        this.client = node.client();
        this.numDocs = fetchNumDocs();
    }

    /**
     * use default cluster name
     * @param indexName
     * @throws Exception
     */
    public ESIndex(String indexName) throws Exception{
        this(indexName,"elasticsearch");
    }

    public ESIndex setLabelField(String labelField) {
        this.labelField = labelField;
        return this;
    }

    public ESIndex setExtLabelField(String extLabelField) {
        this.extLabelField = extLabelField;
        return this;
    }

    public ESIndex setType(String type) {
        this.type = type;
        return this;
    }

    public int getNumDocs() {
        return numDocs;
    }




    /**
     *
     * @param n =1 for now
     * @return
     */
    public Set<Ngram> getNgrams(String id, int n) throws IOException {
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        TermVectorResponse response = client.prepareTermVector(indexName, type, id)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setSelectedFields("body").
                        execute().actionGet();

        Terms terms = response.getFields().terms("body");
        TermsEnum iterator = terms.iterator(null);
        Set<Ngram> ngramSet = new HashSet<>();
        for (int i=0;i<terms.size();i++){
            String term = iterator.next().utf8ToString();
            ngramSet.add(new Ngram(term));
        }

        if(logger.isDebugEnabled()){
            logger.debug("time spent on getNgrams from doc "+id+" = "+stopWatch+
                    " It has "+ngramSet.size()+" ngrams");
        }
        return ngramSet;
    }





    public List<String> getDocs(Ngram ngram, String[] ids) throws Exception{
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }

        /**
         * setSize() has a huge impact on performance, the smaller the faster
         */

        //todo reuse idsFilterBuilder
        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder("document");


        idsFilterBuilder.addIds(ids);

        TermFilterBuilder termFilterBuilder = new TermFilterBuilder("body", ngram.getFeatureName());

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
            logger.debug("time spent on getDocs() for " + ngram.getFeatureName() + " = " + stopWatch+
                    " There are "+list.size()+" matched docs");
        }
        return list;
    }

    public List<String> getDocs(Ngram ngram) throws Exception{
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }

        /**
         * setSize() has a huge impact on performance, the smaller the faster
         */



        TermFilterBuilder termFilterBuilder = new TermFilterBuilder("body", ngram.getFeatureName());

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
            logger.debug("time spent on getDocs() for " + ngram.getFeatureName() + " = " + stopWatch+
                    " There are "+list.size()+" matched docs");
        }
        return list;
    }

    /**
     * experimental
     * @param ngram
     * @return
     * @throws Exception
     */
    public List<String> getDocs2(Ngram ngram) throws Exception{
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }

        /**
         * setSize() has a huge impact on performance, the smaller the faster
         */


        TermQueryBuilder termQueryBuilder = new TermQueryBuilder("body", ngram.getFeatureName());



        SearchResponse response = client.prepareSearch(indexName).setSize(this.numDocs).
                setHighlighterFilter(false).setTrackScores(false).
                setNoFields().setExplain(true).setFetchSource(false).
                setQuery(termQueryBuilder).
                execute().actionGet();
        List<String> list = new ArrayList<>(response.getHits().getHits().length);
        for (SearchHit searchHit : response.getHits()) {
            list.add(searchHit.getId());
            System.out.println(searchHit.getExplanation().toString());
            System.out.println(searchHit.score());
        }
        if(logger.isDebugEnabled()){
            logger.debug("time spent on getDocs() for " + ngram.getFeatureName() + " = " + stopWatch+
                    " There are "+list.size()+" matched docs");
        }
        return list;
    }


    /**
     * only return docs and tfs for docs with ids and have non-zero tfs
     * @param ngram
     * @param ids
     * @return
     * @throws Exception
     */
    public Map<String, Integer> getDocsAndTF(Ngram ngram, String[] ids) throws Exception{
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }

        List<String> docs = getDocs(ngram, ids);
        Map<String, Integer> docsAndTF = new HashMap<>(docs.size());
//        for (Integer docId : docs){
//            String termVectorResponse = getTermVectorResponse("document","body",docId);
//            TermVectorResponsePOJO termVectorResponsePOJO = parseTermVectorResponse(termVectorResponse);
//            int tf = termVectorResponsePOJO.getTF(term);
//            docsAndTF.put(docId,tf);
//        }
        for (String docId : docs){
            docsAndTF.put(docId,1);
        }
        if(logger.isDebugEnabled()) {
            logger.debug("time spent on getDocsAndTF() for "+ngram.getFeatureName()+" = " + stopWatch);
        }
        return docsAndTF;
    }

    public int getLabel(String id){
        GetResponse response = client.prepareGet(indexName, "document", id).setFields(this.labelField)
                .execute()
                .actionGet();
        if (logger.isDebugEnabled()){
            logger.debug("getting label from id "+id+", field "+this.labelField);
        }
        return (int) response.getField(this.labelField).getValue();
    }

    public String getExtLabel(String id){
        GetResponse response = client.prepareGet(indexName, "document", id).setFields(this.extLabelField)
                .execute()
                .actionGet();
        return (String) response.getField(this.extLabelField).getValue();
    }


    public Set<String> listAllFields() throws Exception{
        GetMappingsResponse response = client.admin().indices().prepareGetMappings(this.indexName).
                execute().actionGet();
        MappingMetaData mappingMetaData = response.getMappings().get(this.indexName).get(this.type);
        Map map = (Map)mappingMetaData.getSourceAsMap().get("properties");
        Set<String> fields = new HashSet<>();
        for (Object field: map.keySet()){
            fields.add(field.toString());
        }
        return fields;
    }

    public String getFieldType(String field) {
        GetFieldMappingsResponse response = this.client.admin().indices().
                prepareGetFieldMappings(this.indexName).setTypes(this.type).
                setFields(field).execute().actionGet();
        Map map = (Map)response.mappings().get(this.indexName).get(this.type).
                get(field).sourceAsMap().get(field);
        return map.get("type").toString();
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

    public Map<String,Integer> getTermsAndTfs(String id) throws IOException {
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        TermVectorResponse response = client.prepareTermVector(indexName, type, id)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setTermStatistics(true)
                .setSelectedFields("body").
                        execute().actionGet();

        Terms terms = response.getFields().terms("body");
        TermsEnum iterator = terms.iterator(null);
        Map<String, Integer> termsAndTfs = new HashMap<>();
        for (int i=0;i<terms.size();i++){
            String term = iterator.next().utf8ToString();
            int tf = iterator.docsAndPositions(null,null).freq();
            termsAndTfs.put(term,tf);
        }

        if(logger.isDebugEnabled()){
            logger.debug("time spent on getTermsAndTfs for "+id+" = " + stopWatch);
        }
        return termsAndTfs;
    }

    /**
     * df is from one shard!!!
     * @param id
     * @return
     * @throws IOException
     */
    public Map<String,Float> getTermsAndTfidfs(String id) throws IOException {
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        TermVectorResponse response = client.prepareTermVector(indexName, type, id)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setTermStatistics(true)
                .setSelectedFields("body").
                        execute().actionGet();

        Terms terms = response.getFields().terms("body");
        TermsEnum iterator = terms.iterator(null);
        Map<String, Float> termsAndTfs = new HashMap<>();
        for (int i=0;i<terms.size();i++){
            String term = iterator.next().utf8ToString();
            int tf = iterator.docsAndPositions(null,null).freq();
            int df = iterator.docFreq();
            DefaultSimilarity defaultSimilarity = new DefaultSimilarity();

            /**
             * tf is just tf, not square root of tf as in lucene
             */
            /**
             * df is from lucene
             */
            /** Implemented as <code>log(numDocs/(docFreq+1)) + 1</code>. */
            float tfidf = tf*defaultSimilarity.idf(df,this.numDocs);
            termsAndTfs.put(term,tfidf);
        }

        if(logger.isDebugEnabled()){
            logger.debug("time spent on getTermsAndTfidfs for "+id+" = " + stopWatch);
        }
        return termsAndTfs;
    }

    /**
     * df is from one shard!!!
     * @param id
     * @return
     * @throws IOException
     */
    public Set<NgramInfo> getNgramInfos(String id) throws IOException {
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        TermVectorResponse response = client.prepareTermVector(indexName, type, id)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setTermStatistics(true)
                .setSelectedFields("body").
                        execute().actionGet();

        Terms terms = response.getFields().terms("body");
        TermsEnum iterator = terms.iterator(null);

        Set<NgramInfo> set = new HashSet<>();
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
            Ngram ngram = new Ngram(term);
            NgramInfo ngramInfo = new NgramInfo(ngram);
            ngramInfo.setTf(tf).setDf(df).setTfidf(tfidf);
            set.add(ngramInfo);

        }

        if(logger.isDebugEnabled()){
            logger.debug("time spent on getNgramInfos for "+id+" = " + stopWatch);
        }
        return set;
    }

    /**
     * experimental works but inefficient
     * df is from one shard!!!
     * @param id
     * @return
     * @throws IOException
     */
    public NgramInfo getNgramInfo(String id, Ngram ngram) throws IOException {
        StopWatch stopWatch=null;
        if(logger.isDebugEnabled()){
            stopWatch = new StopWatch();
            stopWatch.start();
        }
        TermVectorResponse response = client.prepareTermVector(indexName, type, id)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setTermStatistics(true)
                .setSelectedFields("body").
                        execute().actionGet();
        BytesRef bytesRef = new BytesRef(ngram.getFeatureName());
        NgramInfo ngramInfo = new NgramInfo(ngram);
        Terms terms = response.getFields().terms("body");
        TermsEnum iterator = terms.iterator(null);
        for (int i=0;i<terms.size();i++){
            iterator.next();
            if (iterator.term().equals(bytesRef)){
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
                ngramInfo.setTf(tf).setDf(df).setTfidf(tfidf);
                break;
            }
        }
        if(logger.isDebugEnabled()){
            logger.debug("time spent on getNgramInfos for "+id+" = " + stopWatch);
        }
        return ngramInfo;
    }

    public void close() {
        this.client.close();
        this.node.close();
    }



    protected String termFilter(String term) throws Exception{
        SearchResponse response = client.prepareSearch(indexName).setSize(10000000).
                addField("").
                setPostFilter(FilterBuilders.termFilter("body", term)).execute().actionGet();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        response.toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject();
        return builder.string();
    }




    protected String getTermVectorResponse(String type, String field, String id) throws Exception {
        TermVectorResponse response = client.prepareTermVector(indexName, type, id)
                .setOffsets(false).setPositions(false).setFieldStatistics(false)
                .setSelectedFields(field).
                        execute().actionGet();
        XContentBuilder builder = XContentFactory.jsonBuilder();
        builder.startObject();
        response.toXContent(builder, ToXContent.EMPTY_PARAMS);
        builder.endObject();
        return builder.string();
    }

    protected int fetchNumDocs() throws Exception{
        SearchResponse response = client.prepareSearch(indexName).setSize(10000000).
                addField("").
                setQuery(QueryBuilders.matchAllQuery()).execute().actionGet();
        //todo safe?
        return (int)response.getHits().totalHits();

    }

    private Object getField(String id, String field){
        GetResponse response = client.prepareGet(this.indexName, this.type, id).
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
//        return getNgrams(n, "body");
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
//            termVectorResponse = getTermVectorResponse("document","body",id);
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
//                setQuery(QueryBuilders.filteredQuery(QueryBuilders.matchAllQuery(),FilterBuilders.termFilter("body",ngram.getFeatureName()).cache(true))).
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
//        IdsFilterBuilder idsFilterBuilder = new IdsFilterBuilder("document");
//
//
//        idsFilterBuilder.addIds(ids);
//
//        TermFilterBuilder termFilterBuilder = new TermFilterBuilder("body", ngram.getFeatureName());
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
//        TermFilterBuilder termFilterBuilder = new TermFilterBuilder("body", ngram.getFeatureName());
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
