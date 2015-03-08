//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.ClfDataSet;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//import edu.neu.ccs.pyramid.util.MathUtil;
//
//import java.io.File;
//import java.util.Arrays;
//import java.util.stream.IntStream;
//
///**
// * check number of ngrams
// * Created by chengli on 1/16/15.
// */
//public class Exp54 {
//    public static void main(String[] args) throws Exception{
//        if (args.length !=1){
//            throw new IllegalArgumentException("please specify the config file");
//        }
//
//        Config config = new Config(args[0]);
//        System.out.println(config);
//        String folder = config.getString("input.folder");
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(new File(folder,"train.trec"), DataSetType.CLF_SPARSE,true);
//        int[] counts = new int[4];
//        IntStream.range(0,dataSet.getNumFeatures())
//                .forEach(i -> {
//                    String name = dataSet.getFeatureSetting(i).getFeatureName();
//                    int length = name.split(" ").length;
//                    if (length <= 3) {
//                        counts[length - 1] += 1;
//                    } else {
//                        counts[3] += 1;
//                    }
//                });
//        double sum = Arrays.stream(counts).sum();
//        if (((int)sum)!=dataSet.getNumFeatures()){
//            throw new RuntimeException("sum)!=dataSet.getNumFeatures()");
//        }
//        System.out.println("total number of featureList = "+dataSet.getNumFeatures());
//        System.out.println("unigram, bigram, trigram, ngram(n>3) counts:");
//        System.out.println(counts[0]+", "+counts[1]+", "+counts[2]+", "+counts[3]);
//        System.out.println("unigram, bigram, trigram, ngram(n>3) ratios:");
//        System.out.println(counts[0]/sum+", "+counts[1]/sum+", "+counts[2]/sum+", "+counts[3]/sum);
//    }
//}
