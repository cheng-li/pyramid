//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.ClfDataSet;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//import edu.neu.ccs.pyramid.elasticsearch.SingleLabelIndex;
//
//import java.util.HashSet;
//import java.util.Set;
//import java.util.stream.IntStream;
//
///**
// * check the difference between two datasets
// * Created by chengli on 1/21/15.
// */
//public class Exp57 {
//    public static void main(String[] args) throws Exception{
//        if (args.length !=1){
//            throw new IllegalArgumentException("please specify the config file");
//        }
//
//        Config config = new Config(args[0]);
//        System.out.println(config);
//
//        ClfDataSet dataSet1 = TRECFormat.loadClfDataSet(config.getString("input.data1"), DataSetType.CLF_SPARSE,true);
//        Set<String> set1 = new HashSet<>();
//        IntStream.range(0,dataSet1.getNumFeatures()).mapToObj(i-> dataSet1.getFeatureSetting(i).getFeatureName())
//                .forEach(set1::add);
//
//
//        ClfDataSet dataSet2 = TRECFormat.loadClfDataSet(config.getString("input.data2"), DataSetType.CLF_SPARSE,true);
//        Set<String> set2 = new HashSet<>();
//        IntStream.range(0,dataSet2.getNumFeatures()).mapToObj(i-> dataSet2.getFeatureSetting(i).getFeatureName())
//                .forEach(set2::add);
//
//
//        Set<String> oneOnly = new HashSet<>(set1);
//        oneOnly.removeAll(set2);
//        System.out.println("only in "+config.getString("input.data1"));
//        System.out.println(oneOnly);
//
//
//        Set<String> twoOnly = new HashSet<>(set2);
//        oneOnly.removeAll(set1);
//        System.out.println("only in "+config.getString("input.data2"));
//        System.out.println(twoOnly);
//
//
//
//
//    }
//}
