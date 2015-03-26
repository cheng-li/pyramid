package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.classification.PriorProbClassifier;
import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.Ngram;
import edu.neu.ccs.pyramid.regression.regression_tree.SplitResult;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.mahout.math.Vector;
import org.elasticsearch.action.ListenableActionFuture;
import org.elasticsearch.action.search.MultiSearchRequestBuilder;
import org.elasticsearch.action.search.SearchRequestBuilder;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.index.query.QueryBuilders;

import java.io.File;
import java.util.*;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * elasticsearch benchmark
 * Created by chengli on 3/25/15.
 */
public class Exp76 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);
        check(config);



    }

    private static void check(Config config) throws Exception{
        String input = config.getString("input.folder");
        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(input, "train.trec"),
                DataSetType.CLF_SPARSE, true);
        ESIndex index = loadIndex(config);
        countSearch(dataSet,index);
        benchIntersection(dataSet,index);
        smartSearch(dataSet,index);
        completeSearch(dataSet,index);

    }

    private static void completeSearch(DataSet dataSet, ESIndex index){
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        List<String> ngrams = dataSet.getFeatureList().getAll().stream().map(feature -> ((Ngram)feature).getNgram())
                .collect(Collectors.toList());
        ngrams.parallelStream().forEach(ngram -> index.matchPhrase(index.getBodyField(),ngram,0));
        System.out.println("time spent on complete search = "+stopWatch);
    }

    private static void smartSearch(DataSet dataSet, ESIndex index) throws Exception{
        ForkJoinPool pool = new ForkJoinPool(90);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        IdTranslator idTranslator = dataSet.getIdTranslator();
        Map<String,Integer> ngramIndexMap = new HashMap<>();
        for (Feature feature: dataSet.getFeatureList().getAll()){
            int featureIndex = feature.getIndex();
            String ngram = ((Ngram)feature).getNgram();
            ngramIndexMap.put(ngram,featureIndex);
        }
        List<String> ngrams = dataSet.getFeatureList().getAll().stream().map(feature -> ((Ngram)feature).getNgram())
                .collect(Collectors.toList());
        pool.submit(() -> ngrams.parallelStream().forEach(ngram -> {
            Set<Integer> intersection = intersection(dataSet, ngram, ngramIndexMap);
            String[] ids = intersection.stream().map(idTranslator::toExtId).toArray(String[]::new);
            index.matchPhrase(index.getBodyField(), ngram, ids, 0);
        })).get();

        System.out.println("time spent on smart search = " + stopWatch);
    }

    private static void countSearch(DataSet dataSet, ESIndex index) throws Exception{
        ForkJoinPool pool = new ForkJoinPool(90);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        IdTranslator idTranslator = dataSet.getIdTranslator();
        Map<String,Integer> ngramIndexMap = new HashMap<>();
        for (Feature feature: dataSet.getFeatureList().getAll()){
            int featureIndex = feature.getIndex();
            String ngram = ((Ngram)feature).getNgram();
            ngramIndexMap.put(ngram,featureIndex);
        }
        List<String> ngrams = dataSet.getFeatureList().getAll().stream().map(feature -> ((Ngram)feature).getNgram())
                .collect(Collectors.toList());
        pool.submit(() -> ngrams.parallelStream().forEach(ngram -> {
            int featureIndex = ngramIndexMap.get(ngram);
            int size = dataSet.getColumn(featureIndex).getNumNonZeroElements();
            SearchResponse response = index.getClient().prepareSearch(index.getIndexName()).setSize(size).
                    setHighlighterFilter(false).setTrackScores(false).
                    setNoFields().setExplain(false).setFetchSource(false).
                    setQuery(QueryBuilders.matchPhraseQuery(index.getBodyField(), ngram).slop(0)
                            .analyzer("whitespace")).
                    execute().actionGet();
        })).get();

        System.out.println("time spent on count search = " + stopWatch);
    }

    private static void benchIntersection(DataSet dataSet, ESIndex index) throws Exception{
        ForkJoinPool pool = new ForkJoinPool(900);
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        IdTranslator idTranslator = dataSet.getIdTranslator();
        Map<String,Integer> ngramIndexMap = new HashMap<>();
        for (Feature feature: dataSet.getFeatureList().getAll()){
            int featureIndex = feature.getIndex();
            String ngram = ((Ngram)feature).getNgram();
            ngramIndexMap.put(ngram,featureIndex);
        }
        List<String> ngrams = dataSet.getFeatureList().getAll().stream().map(feature -> ((Ngram)feature).getNgram())
                .collect(Collectors.toList());
        pool.submit(() -> ngrams.parallelStream().forEach(ngram -> {
            Set<Integer> intersection = intersection(dataSet, ngram, ngramIndexMap);
            String[] ids = intersection.stream().map(idTranslator::toExtId).toArray(String[]::new);

        })).get();

        System.out.println("time spent on intersections = " + stopWatch);
    }



    static SingleLabelIndex loadIndex(Config config) throws Exception{
        SingleLabelIndex.Builder builder = new SingleLabelIndex.Builder()
                .setIndexName(config.getString("index.indexName"))
                .setClusterName(config.getString("index.clusterName"))
                .setClientType(config.getString("index.clientType"))
                .setLabelField(config.getString("index.labelField"))
                .setExtLabelField(config.getString("index.extLabelField"))
                .setDocumentType(config.getString("index.documentType"));
        if (config.getString("index.clientType").equals("transport")){
            String[] hosts = config.getString("index.hosts").split(Pattern.quote(","));
            String[] ports = config.getString("index.ports").split(Pattern.quote(","));
            builder.addHostsAndPorts(hosts,ports);
        }
        SingleLabelIndex index = builder.build();
        System.out.println("index loaded");
        System.out.println("there are "+index.getNumDocs()+" documents in the index.");
//        for (int i=0;i<index.getNumDocs();i++){
//            System.out.println(i);
//            System.out.println(index.getLabel(""+i));
//        }
        return index;
    }

    private static Set<Integer> intersection(DataSet dataSet, String ngram, Map<String, Integer> ngramIndexMap){
        String[] unigrams = ngram.split(" ");
        List<Set<Integer>> sets = Arrays.stream(unigrams).map(unigram -> {
            //todo why happen?
            if (!ngramIndexMap.containsKey(unigram)) {
                return new HashSet<Integer>();
            }
            int index = ngramIndexMap.get(unigram);
            Vector vector = dataSet.getColumn(index);
            Set<Integer> set = new HashSet<Integer>();
            for (Vector.Element element : vector.nonZeroes()) {
                set.add(element.index());
            }
            return set;
        }).collect(Collectors.toList());
        Set<Integer> intersection = sets.get(0);
        for (Set<Integer> set: sets){
            intersection.retainAll(set);
        }
        return intersection;
    }
}
