package edu.neu.ccs.pyramid.feature_extraction;

import edu.neu.ccs.pyramid.elasticsearch.ESIndex;
import edu.neu.ccs.pyramid.elasticsearch.ESIndexBuilder;


import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;


public class PhraseDetectorTest {
    public static void main(String[] args) throws Exception{
        test8();

    }

//    static void test1() throws Exception {
//        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
//                .build();
//        Map<Integer,String> map = index.getTermVector("0");
//
//
//        index.close();
//        System.out.println(map);
//        System.out.println(PhraseDetector.getPhrases(map, 11));
//    }
//
    static void test2() throws Exception {
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        Map<Integer,String> map = index.getTermVector("1");



//        System.out.println(map);
        DFStats dfStats = DFStats.deserialize("/Users/chengli/tmp/dfstats.ser");
        Set<String> references = new HashSet<>();
        List<DFStat> list = dfStats.getSortedDFStats(0,20);
        list.stream().limit(100).forEach(dfStat -> references.add(dfStat.getPhrase()));
        System.out.println(references);
        PhraseDetector phraseDetector = new PhraseDetector(index).setMinDf(20);
        System.out.println(phraseDetector.getPhraseInfos(map, references));
        index.close();
    }

    static void test3() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        Map<Integer,String> map = index.getTermVector("231");
        System.out.println(map);
        PhraseDetector detector = new PhraseDetector(index);
        detector.setMinDf(9);
        System.out.println(detector.exploreLeft(map, 28));
        index.close();

    }

    static void test4() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        Map<Integer,String> map = index.getTermVector("648");
        System.out.println(map);
        PhraseDetector detector = new PhraseDetector(index);
        detector.setMinDf(20);
        System.out.println(detector.exploreLeft(map, 90));
        index.close();

    }

    static void test5() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        Map<Integer,String> map = index.getTermVector("231");
        System.out.println(map);
        PhraseDetector detector = new PhraseDetector(index);
        detector.setMinDf(9);
        System.out.println(detector.exploreRight(map, 26));
        index.close();

    }

    static void test6() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        Map<Integer,String> map = index.getTermVector("648");
        System.out.println(map);
        PhraseDetector detector = new PhraseDetector(index);
        detector.setMinDf(20);
        System.out.println(detector.exploreRight(map, 89));
        index.close();

    }

    static void test7() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        Map<Integer,String> map = index.getTermVector("231");
        System.out.println(map);
        PhraseDetector detector = new PhraseDetector(index);
        detector.setMinDf(9);
        List<PhraseInfo> left = detector.exploreLeft(map, 27);
        System.out.println("left");
        System.out.println(left);
        List<PhraseInfo> right = detector.exploreRight(map,27);
        System.out.println("right");
        System.out.println(right);
        List<PhraseInfo> connected = detector.connect(left, right);
        System.out.println("connected");
        System.out.println(connected);
        index.close();

    }

    static void test8() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        Map<Integer,String> map = index.getTermVector("231");
        System.out.println(map);
        PhraseDetector detector = new PhraseDetector(index);
        detector.setMinDf(9);
        Set<PhraseInfo> all = detector.getPhraseInfos(map,27);
        System.out.println(all);
        index.close();

    }

    static void test9() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        Map<Integer,String> map = index.getTermVector("231");
        System.out.println(map);
        PhraseDetector detector = new PhraseDetector(index);
        detector.setMinDf(9);
        Set<PhraseInfo> all = detector.getPhraseInfos(map,26);
        System.out.println(all);
        index.close();

    }

    static void test10() throws Exception{
        ESIndex index = ESIndexBuilder.builder().setClientType("node").setIndexName("cnn")
                .build();
        Map<Integer,String> map = index.getTermVector("231");
        System.out.println(map);
        PhraseDetector detector = new PhraseDetector(index);
        detector.setMinDf(9);
        Set<PhraseInfo> all = detector.getPhraseInfos(map,28);
        System.out.println(all);
        index.close();

    }

}