package edu.neu.ccs.pyramid.experiment;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.data_formatter.eightnewsgroup.Merger;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.util.DirWalker;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * selection top columns of 20nesgroup feature matrix,
 * merge into 8 classes
 * Created by chengli on 10/5/14.
 */
public class Exp9 {
    public static void main(String[] args) throws Exception{
        if (args.length !=1){
            throw new IllegalArgumentException("please specify the config file");
        }

        Config config = new Config(args[0]);
        System.out.println(config);

        ClfDataSet inputData = loadInput(config);
        List<Integer> featues = loadFeatureList(config);
        List<Integer> filteredFeatures = filter(featues,inputData);
        ClfDataSet trimmed = trim(inputData,filteredFeatures);
        ClfDataSet merged = mergeLabels(trimmed);
        saveNew(config,merged);
    }

    private static ClfDataSet loadInput(Config config) throws Exception{
        String input = config.getString("input.dataset");
        return TRECFormat.loadClfDataSet(input, DataSetType.CLF_SPARSE,true);
    }

    private static List<Integer> loadFeatureList(Config config) throws Exception{
        String folder = config.getString("input.featureFolder");
        List<File> files = DirWalker.getFiles(folder).stream()
                .filter(file -> file.getName().endsWith(".indices"))
                .collect(Collectors.toList());
        //remove duplicates
        Set<Integer> indices = new HashSet<>();
        for (File file: files){
            indices.addAll(loadOneFeatureFile(file));
        }

        List<Integer> indicesList = new ArrayList<>();
        indicesList.addAll(indices);
        //make sure the order is unique
        Collections.sort(indicesList);
        return indicesList;
    }

    private static Set<Integer> loadOneFeatureFile(File file) throws Exception{
        Set<Integer> set = new HashSet<>();
        try(BufferedReader br = new BufferedReader(new FileReader(file))
        ){
          String line = null;
          while((line=br.readLine())!=null){
              set.add(Integer.parseInt(line));
          }
        }
        return set;
    }

    /**
     * get rid of email addresses
     * @param indices
     * @param dataSet
     * @return
     */
    private static List<Integer> filter(List<Integer> indices, ClfDataSet dataSet){
        return indices.stream()
                .filter(i -> !(dataSet.getFeatureColumn(i).getSetting().getFeatureName().split(Pattern.quote(".")).length >= 3))
                .filter(i -> !(dataSet.getFeatureColumn(i).getSetting().getFeatureName().endsWith(".com")))
                .collect(Collectors.toList());
    }

    private static ClfDataSet trim(ClfDataSet input, List<Integer> toKeep){
        return  DataSetUtil.trim(input,toKeep);
    }

    private static void saveNew(Config config, ClfDataSet newData) throws Exception{
        String saveTo = config.getString("archive.dataset");
        TRECFormat.save(newData,saveTo);
        DataSetUtil.dumpDataSettings(newData,new File(config.getString("archive.dataset"),"data_settings.txt"));
        DataSetUtil.dumpFeatureSettings(newData,new File(config.getString("archive.dataset"),"feature_settings.txt"));
    }

    private static ClfDataSet mergeLabels(ClfDataSet dataSet){
        ClfDataSet merged = DataSetUtil.changeLabels(dataSet,8);
        int[] oldLabels = dataSet.getLabels();
        for (int i=0;i<dataSet.getNumDataPoints();i++){
            int newLabel = Merger.merged(oldLabels[i]);
            merged.setLabel(i,newLabel);
        }
        DataSetUtil.setExtLabels(merged,Merger.extLabels);
        return merged;
    }

}
