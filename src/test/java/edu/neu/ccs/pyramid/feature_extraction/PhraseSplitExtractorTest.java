package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.elasticsearch.ESIndex;


import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.*;

public class PhraseSplitExtractorTest {
    public static void main(String[] args) throws Exception{
//        test1();

    }

//    static void test1() throws Exception{
//        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
//                .build();
//        Map<Integer,String> map = index.getTermVector("0");
//
//
//
////        System.out.println(map);
//        DFStats dfStats = DFStats.deserialize("/Users/chengli/tmp/dfstats.ser");
//        Set<String> references = new HashSet<>();
//        List<DFStat> list = dfStats.getSortedDFStats(0,20);
//        list.stream().limit(100).forEach(dfStat -> references.add(dfStat.getPhrase()));
//        System.out.println(references);
//        Set<String> phrases = PhraseDetector.getPhrases(map, references);
//        PhraseSplitExtractor extractor = new PhraseSplitExtractor(index,null);
//        Set<String> blackList = new HashSet<>();
//        System.out.println(extractor.filter(phrases,blackList));
//        index.close();
//    }



}