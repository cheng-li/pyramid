//package edu.neu.ccs.pyramid.experiment;
//
//import edu.neu.ccs.pyramid.configuration.Config;
//import edu.neu.ccs.pyramid.dataset.ClfDataSet;
//import edu.neu.ccs.pyramid.dataset.DataSetType;
//import edu.neu.ccs.pyramid.dataset.TRECFormat;
//
//import java.io.*;
//import java.util.ArrayList;
//import java.util.List;
//
///**
// * given feature indices, get feature names
// * Created by chengli on 12/12/14.
// */
//public class Exp32 {
//    public static void main(String[] args) throws Exception{
//        if (args.length !=1){
//            throw new IllegalArgumentException("Please specify a properties file.");
//        }
//
//        Config config = new Config(args[0]);
//        System.out.println(config);
//
//        File folder = new File(config.getString("input.folder"));
//        File trec = new File(folder,"train.trec");
//        ClfDataSet dataSet = TRECFormat.loadClfDataSet(trec, DataSetType.CLF_SPARSE,true);
//        String featureIndicesFile = config.getString("featureIndicesFile");
//        String line;
//        try (BufferedReader br = new BufferedReader(new FileReader(featureIndicesFile))){
//            line = br.readLine();
//        };
//
//        String[] indices = line.split(" ");
//        int top = config.getInt("top");
//        List<String> terms = new ArrayList<>();
//        for (int i=0;i<top;i++){
//            int index = Integer.parseInt(indices[i]);
//            String term = dataSet.getFeatureSetting(index).getFeatureName();
//            terms.add(term);
//        }
//
//        String outputFolder = config.getString("output.folder");
//        File output = new File(outputFolder,"initialSeeds");
//        try( BufferedWriter bw = new BufferedWriter(new FileWriter(output))){
//            for (String term: terms){
//                bw.write(term);
//                bw.newLine();
//            }
//        }
//    }
//}
