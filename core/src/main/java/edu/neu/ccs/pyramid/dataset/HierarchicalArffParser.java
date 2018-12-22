package edu.neu.ccs.pyramid.dataset;

import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class HierarchicalArffParser {
    public static long parseNumFeatures(File file) throws Exception{
        return FileUtils.readLines(file).stream().filter(line->
                isFeatureLine(line)).count();
    }

    public static LabelTranslator loadLabelTranslator(File file) throws Exception{
        String def = FileUtils.readLines(file).stream().filter(line->isLabelLine(line)).findFirst().get();
        String part = def.split("\\s+")[3];
        String[] labels = part.split(",");
        LabelTranslator labelTranslator = new LabelTranslator(labels);
        return labelTranslator;
    }

    private static boolean isFeatureLine(String line){
        String[] split = line.split("\\s+");
        return split[0].equalsIgnoreCase("@attribute")&&(!split[1].equalsIgnoreCase("class"));
    }


    private static boolean isLabelLine(String line){
        String[] split = line.split("\\s+");
        return split[0].equalsIgnoreCase("@attribute")&&(split[1].equalsIgnoreCase("class"));
    }

    public static List<String> getParents(String label){
        List<String> parents = new ArrayList<>();
        String[] split = label.split(Pattern.quote("/"));
        if (split.length>1){
            String parent = split[0];
            parents.add(parent);
            for (int i=1;i<split.length-1;i++){
                parent = parent+"/"+split[i];
                parents.add(parent);
            }
        }
        return parents;
    }

    public static long parseNumInstances(File file) throws Exception{
        return FileUtils.readLines(file).stream().filter(line->line.startsWith("{")&&line.endsWith("}")).count();
    }

    public static MultiLabelClfDataSet load(File file) throws Exception{
        int numInstances = (int)parseNumInstances(file);
        int numFeatures = (int)parseNumFeatures(file);
        LabelTranslator labelTranslator = loadLabelTranslator(file);
        int numLabels = labelTranslator.getNumClasses();
        MultiLabelClfDataSet dataSet = MLClfDataSetBuilder.getBuilder().numDataPoints(numInstances)
                .numFeatures(numFeatures).numClasses(numLabels).density(Density.SPARSE_RANDOM).build();
        dataSet.setLabelTranslator(labelTranslator);
        loadFeatureValues(dataSet,file);
        return dataSet;
    }

    public static void loadFeatureValues(MultiLabelClfDataSet dataSet, File file)throws Exception{
        List<String> lines = FileUtils.readLines(file).stream().filter(line->line.startsWith("{")&&line.endsWith("}"))
                .collect(Collectors.toList());
        for (int i=0;i<lines.size();i++){
            String line = lines.get(i).replace("{","").replace("}","");
            String[] split = line.split(",");
            for (String s: split){
                String[] ss = s.split(" ");
                int index = Integer.parseInt(ss[0]);
                if (index<=dataSet.getNumFeatures()){
                    double value = Double.parseDouble(ss[1]);
                    dataSet.setFeatureValue(i,index-1,value);
                } else {
                    // label
                    String[] labels = ss[1].split("@");
                    for (String label: labels){
                        if (!label.isEmpty()){
                            System.out.println("i="+i);
                            System.out.println("label="+label);
                            dataSet.addLabel(i,dataSet.getLabelTranslator().toIntLabel(label));
                        }

                    }
                }
            }
        }
    }

    public static void toTrec(String inputFile, String outputFile) throws Exception{
        MultiLabelClfDataSet dataSet = load(new File(inputFile));
        TRECFormat.save(dataSet,outputFile);
        Hierarchy hierarchy = new Hierarchy(dataSet.getNumClasses());
        for (int l=0;l<dataSet.getNumClasses();l++){
            String label = dataSet.getLabelTranslator().toExtLabel(l);
            List<String> parents = getParents(label);
            List<Integer> intP = parents.stream().map(p->dataSet.getLabelTranslator().toIntLabel(p)).collect(Collectors.toList());
            hierarchy.setParentsForLabel(l,intP);
        }

        Serialization.serialize(hierarchy,new File(outputFile,"hierarchy.ser"));
        FileUtils.writeStringToFile(new File(outputFile,"hierarchy.txt"),hierarchy.toString());

    }
}
