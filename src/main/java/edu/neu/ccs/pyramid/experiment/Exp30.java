package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.TRECFormat;
import edu.neu.ccs.pyramid.util.Pair;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

/**
 * given logistic regression coefficients, print top featureList
 * Created by chengli on 11/23/14.
 */
public class Exp30 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        ClfDataSet dataSet = loadDataSet(config);
        List<Pair<Integer,Double>> indexWeightPairs = loadIndexWeightPairs(config);
        List<Pair<String, Double>> nameWeightPairs = loadNameWeightPairs(dataSet,indexWeightPairs);


        showTopFeatures(config,nameWeightPairs);
        showPosFeatures(config,nameWeightPairs);
        showNegFeatures(config,nameWeightPairs);
        showTopUnigramFeatures(config,nameWeightPairs);
        showPosUnigramFeatures(config,nameWeightPairs);
        showNegUnigramFeatures(config,nameWeightPairs);
        showTopNgramFeatures(config,nameWeightPairs);
        showPosNgramFeatures(config,nameWeightPairs);
        showNegNgramFeatures(config,nameWeightPairs);


    }

    private static void showTopFeatures(Config config, List<Pair<String, Double>> nameWeightPairs){
        int top = config.getInt("top");
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<String> topFeatures = nameWeightPairs.stream().sorted(comparator.reversed())
                .map(Pair::getFirst).limit(top).collect(Collectors.toList());
        System.out.println("top featureList ");
        System.out.println(topFeatures);
    }

    private static void showTopUnigramFeatures(Config config, List<Pair<String, Double>> nameWeightPairs){
        int top = config.getInt("top");
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<String> topFeatures = nameWeightPairs.stream()
                .filter(pair -> pair.getFirst().split(" ").length==1)
                .sorted(comparator.reversed())
                .map(Pair::getFirst)
                .limit(top).collect(Collectors.toList());

        System.out.println("top unigram featureList ");
        System.out.println(topFeatures);
    }

    private static void showTopNgramFeatures(Config config, List<Pair<String, Double>> nameWeightPairs){
        int top = config.getInt("top");
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<String> topFeatures = nameWeightPairs.stream()
                .filter(pair -> pair.getFirst().split(" ").length>1)
                .sorted(comparator.reversed())
                .map(Pair::getFirst)
                .limit(top).collect(Collectors.toList());

        System.out.println("top ngram featureList ");
        System.out.println(topFeatures);
    }



    private static void showPosFeatures(Config config, List<Pair<String, Double>> nameWeightPairs){
        int top = config.getInt("top");
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<String> topFeatures = nameWeightPairs.stream().filter(pair -> pair.getSecond()>0)
                .sorted(comparator.reversed())
                .map(Pair::getFirst).limit(top).collect(Collectors.toList());
        System.out.println("top positive featureList ");
        System.out.println(topFeatures);
    }

    private static void showNegFeatures(Config config, List<Pair<String, Double>> nameWeightPairs){
        int top = config.getInt("top");
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<String> topFeatures = nameWeightPairs.stream().filter(pair -> pair.getSecond()<0)
                .sorted(comparator.reversed())
                .map(Pair::getFirst).limit(top).collect(Collectors.toList());
        System.out.println("top negative featureList ");
        System.out.println(topFeatures);
    }

    private static void showPosUnigramFeatures(Config config, List<Pair<String, Double>> nameWeightPairs){
        int top = config.getInt("top");
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<String> topFeatures = nameWeightPairs.stream().filter(pair -> pair.getSecond()>0)
                .filter(pair -> pair.getFirst().split(" ").length==1)
                .sorted(comparator.reversed())
                .map(Pair::getFirst).limit(top).collect(Collectors.toList());
        System.out.println("top positive unigram featureList ");
        System.out.println(topFeatures);
    }

    private static void showNegUnigramFeatures(Config config, List<Pair<String, Double>> nameWeightPairs){
        int top = config.getInt("top");
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<String> topFeatures = nameWeightPairs.stream().filter(pair -> pair.getSecond()<0)
                .filter(pair -> pair.getFirst().split(" ").length==1)
                .sorted(comparator.reversed())
                .map(Pair::getFirst).limit(top).collect(Collectors.toList());
        System.out.println("top negative unigram featureList ");
        System.out.println(topFeatures);
    }

    private static void showPosNgramFeatures(Config config, List<Pair<String, Double>> nameWeightPairs){
        int top = config.getInt("top");
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<String> topFeatures = nameWeightPairs.stream().filter(pair -> pair.getSecond()>0)
                .filter(pair -> pair.getFirst().split(" ").length>1)
                .sorted(comparator.reversed())
                .map(Pair::getFirst).limit(top).collect(Collectors.toList());
        System.out.println("top positive ngram featureList ");
        System.out.println(topFeatures);
    }

    private static void showNegNgramFeatures(Config config, List<Pair<String, Double>> nameWeightPairs){
        int top = config.getInt("top");
        Comparator<Pair<String,Double>> comparator = Comparator.comparing(pair -> Math.abs(pair.getSecond()));
        List<String> topFeatures = nameWeightPairs.stream().filter(pair -> pair.getSecond()<0)
                .filter(pair -> pair.getFirst().split(" ").length>1)
                .sorted(comparator.reversed())
                .map(Pair::getFirst).limit(top).collect(Collectors.toList());
        System.out.println("top negative ngram featureList ");
        System.out.println(topFeatures);
    }

    private static ClfDataSet loadDataSet(Config config) throws Exception{
        File trecFile = new File(config.getString("input.folder"),
                config.getString("input.trainData"));
        ClfDataSet clfDataSet = TRECFormat.loadClfDataSet(trecFile, DataSetType.CLF_SPARSE, true);
        return clfDataSet;
    }

    private static List<Pair<Integer,Double>> loadIndexWeightPairs(Config config) throws Exception{
        List<Pair<Integer,Double>> pairs = new ArrayList<>();
        File coeffFile = new File(config.getString("input.coefficients"));
        try(BufferedReader br = new BufferedReader(new FileReader(coeffFile))
        ){
            String line = null;
            int featureIndex = 0;
            while ((line=br.readLine())!=null){
                double coeff = Double.parseDouble(line);
                Pair<Integer,Double> pair = new Pair<>(featureIndex,coeff);
                featureIndex += 1;
                pairs.add(pair);
            }
        }
        return pairs;
    }

    private static List<Pair<String, Double>> loadNameWeightPairs(ClfDataSet dataSet, List<Pair<Integer,Double>> indexWeightPairs){
        List<Pair<String, Double>> nameWeightPairs = new ArrayList<>();
        for (int i=0;i<indexWeightPairs.size();i++){
            int index = indexWeightPairs.get(i).getFirst();
            String name = dataSet.getFeatureList().get(index).getName();
            double weight = indexWeightPairs.get(i).getSecond();
            Pair<String, Double> pair = new Pair<>(name,weight);
            nameWeightPairs.add(pair);
        }
        return nameWeightPairs;
    }

    private static List<String> getFeatureNames(ClfDataSet dataSet, List<Integer> featureIndices){
        if (featureIndices.size()!=dataSet.getNumFeatures()){
            throw new RuntimeException("featureIndices.size()!=dataSet.getNumFeatures()");
        }
        return featureIndices.parallelStream().map(i -> dataSet.getFeatureList().get(i).getName())
                .collect(Collectors.toList());
    }
}
