package edu.neu.ccs.pyramid.util;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.LabelTranslator;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.FMeasure;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class ReportUtils {



    public static void generateBreakDownReport(Config config, Map<String,String> idPc1, String ensembleReportPath, List<String> testDocIds, double ctat_clipped,String pathToSaveReport)throws Exception{


        StringBuilder sb = new StringBuilder();
        sb.append("Tenant").append("\t").append("procedure category").append("\t").append("CTAT").append("\t")
                .append("total docs").append("\t").append("auto percent").append("\t").append("auto accuracy");
        List<String> modelNames = config.getStrings("modelNames");
        for(int i = 0; i < modelNames.size(); i++){

            sb.append("\t").append(modelNames.get(i));

        }
        sb.append("\n");

        Set<String> pcSet = idPc1.entrySet().stream().map(entry->entry.getValue()).collect(Collectors.toSet());
        List<String> pcList = pcSet.stream().sorted().collect(Collectors.toList());

        for(int i = 0; i < pcList.size(); i++){
            String category = pcList.get(i);
            sb.append(config.getString("tenantName")).append("\t").append(category).append("\t").append(ctat_clipped).append("\t");
            List<String> categoryLinesDocIds = testDocIds.stream().filter(id->idPc1.get(id).equals(category)).collect(Collectors.toList());
            sb.append(categoryLinesDocIds.size()).append("\t");
            getAutocodingPercentPerCategory(categoryLinesDocIds,ensembleReportPath,modelNames,ctat_clipped,sb);

        }

        FileUtils.writeStringToFile(new File(pathToSaveReport),sb.toString());


    }


    public static void getAutocodingPercentPerCategory(List<String> categoryLinesDocIds, String ensembleReportPath, List<String> modelNames, double ctat_clipped, StringBuilder sb)throws Exception{

        double automated = 0.0;
        double correctAuto = 0.0;

        List<String> lines = FileUtils.readLines(new File(ensembleReportPath));
        Set<String> set = new HashSet<>(categoryLinesDocIds);
        List<String> desiredLines = lines.stream().filter(line->set.contains(line.split("\t")[0])).filter(line->line.split("\t")[2].equals("set")).collect(Collectors.toList());
        for(int i = 0; i < desiredLines.size(); i++){
            String[] oneLine = desiredLines.get(i).split("\t");
            if(Double.parseDouble(oneLine[3])>=ctat_clipped){
                automated += 1;
                if(Double.parseDouble(oneLine[4])==1.0){
                    correctAuto += 1;
                }
            }

        }

        sb.append((automated*1.0)/desiredLines.size()).append("\t").append((correctAuto*1.0)/automated).append("\t");
        for(int i = 0; i < modelNames.size(); i++){
            final int index = i;
            sb.append (desiredLines.stream().filter(line->line.split("\t")[9].equals(modelNames.get(index))).count()).append("\t");

        }

        sb.append("\n");


    }



    public static List<String> getTestSetInfoFromAllRec(String allRecPath, List<String> testDocIds)throws Exception{

        Set<String> setDocIds = new HashSet<>(testDocIds);

        return Files.lines(Paths.get(allRecPath)).filter(line->setDocIds.contains(line.split("\t")[1])).collect(Collectors.toList());

    }



    public static List<String> getDocIds(String reportFile)throws Exception{

        List<String> lines = FileUtils.readLines(new File(reportFile));
        return lines.stream().filter(line->line.split("\t")[2].equals("set")).map(line->line.split("\t")[0]).collect(Collectors.toList());

    }






    public static List<Pair<Double, Double>> getConfidenceCorrectness(String reportPath)throws Exception{

        List<String> lines = FileUtils.readLines(new File(reportPath));
        List<String> filteredLines = lines.stream().filter(line->line.split("\t")[2].equals("set")).collect(Collectors.toList());
        List<Pair<Double,Double>> list = new ArrayList<>();

        for(int i = 0; i < filteredLines.size(); i++){
            Pair<Double,Double> pair = new Pair<>();
            String[] line = filteredLines.get(i).split("\t");

            pair.setFirst(Double.parseDouble(line[3]));
            pair.setSecond(Double.parseDouble(line[4]));
            list.add(pair);

        }

       return list;

    }





    public static Map<String,String> getIDPrediction(String ReportPath)throws Exception{

        List<String> lines = FileUtils.readLines(new File(ReportPath));
        List<String> filteredLines = lines.stream().filter(line->line.split("\t")[2].equals("set")).collect(Collectors.toList());
        Map<String,String> map = new HashMap<>();
        for(int i = 0; i < filteredLines.size(); i++){
            String[] line = filteredLines.get(i).split("\t");
            map.put(line[0],line[1]);
        }

        return map;

    }


    public static Map<String,String> getIDGroundTruth(MultiLabelClfDataSet dataSet)throws Exception{

        Map<String,String> map = new HashMap<>();

        for (int i = 0;i < dataSet.getNumDataPoints();i++) {
            String docID = dataSet.getIdTranslator().toExtId(i);
            String groundTruth = dataSet.getMultiLabels()[i].toStringWithExtLabels(dataSet.getLabelTranslator()).split("\\[")[1].split("\\]")[0];
            map.put(docID,groundTruth);

        }

        return map;

    }

    public static Map<String,Double> getIDConfidence(String reportFile)throws Exception{

        List<String> lines = FileUtils.readLines(new File(reportFile));
        List<String> filteredLines = lines.stream().filter(line->line.split("\t")[2].equals("set")).collect(Collectors.toList());
        Map<String, Double> map = new HashMap<>();
       for(int i = 0; i < filteredLines.size(); i++){
           String[] info = filteredLines.get(i).split("\t");
           map.put(info[0],Double.parseDouble(info[3]));
       }
       return map;

    }

    public static double[] getConfidence(MultiLabelClfDataSet dataSet, String reportFile)throws Exception{

        Map<String, Double> map = getIDConfidence(reportFile);
        double[] confidence = new double[dataSet.getNumDataPoints()];

        for(int i = 0; i < dataSet.getNumDataPoints(); i++){
            String docID = dataSet.getIdTranslator().toExtId(i);
            confidence[i] = map.get(docID);
        }

        return confidence;


    }


    public static Map<String,List<String>> getIDPredictionConfidence(String reportPath) throws Exception{

        List<String> lines = FileUtils.readLines(new File(reportPath));
        List<String> filteredLines = lines.stream().filter(line->line.split("\t")[2].equals("set")).collect(Collectors.toList());
        Map<String,List<String>> map = new HashMap<>();
        for(int i =0;i<filteredLines.size();i++){
            String[] line = filteredLines.get(i).split("\t");
            String id = line[0];
            map.put(id, new ArrayList<>());
            map.get(id).add(line[1]);
            map.get(id).add(line[3]);


        }

        return map;

    }


    public static List<Pair<Double,Double>> computeConfidenceF1(MultiLabelClfDataSet dataSet, String reportPath)throws Exception {

        Map<String, List<String>> idPredictionConfidence = ReportUtils.getIDPredictionConfidence(reportPath);
        List<Pair<Double, Double>> list = new ArrayList<>();
        for (int i = 0; i < dataSet.getNumDataPoints(); i++) {
            String docID = dataSet.getIdTranslator().toExtId(i);
            MultiLabel groundTruth = dataSet.getMultiLabels()[i];
            Pair<Double, Double> pair = new Pair<>();
            pair.setFirst(Double.parseDouble(idPredictionConfidence.get(docID).get(1)));
            String[] prediction = idPredictionConfidence.get(docID).get(0).split(",");
            MultiLabel pre = new MultiLabel();
            for (int j = 0; j < prediction.length; j++) {
                pre.addLabel(dataSet.getLabelTranslator().toIntLabel(prediction[j]));
            }

            pair.setSecond(FMeasure.f1(pre, groundTruth));

            list.add(pair);

        }

        return list;
    }

    public static List<Pair<Double,Double>> getConfidenceF1(String reportPath)throws Exception {
        List<String> lines = FileUtils.readLines(new File(reportPath));
        List<String> filteredLines = lines.stream().filter(line->line.split("\t")[2].equals("set")).collect(Collectors.toList());
        List<Pair<Double,Double>> list = new ArrayList<>();
        for(int i = 0; i < filteredLines.size(); i++){
            String[] lineInfo = filteredLines.get(i).split("\t");
            Pair<Double,Double> pair = new Pair<>();
            pair.setFirst(Double.parseDouble(lineInfo[3]));
            pair.setSecond(Double.parseDouble(lineInfo[8]));
            list.add(pair);

        }

        return list;



    }







    public static MultiLabel[] getMultilabelTypePredictions(MultiLabelClfDataSet dataSet, LabelTranslator labelTranslator, String reportPath)throws Exception{

        Map<String,String> idPredictions = getIDPrediction(reportPath);

        MultiLabel[] predictions = new MultiLabel[dataSet.getNumDataPoints()];

        for(int i = 0; i < dataSet.getNumDataPoints(); i++) {
            String docID = dataSet.getIdTranslator().toExtId(i);
            MultiLabel pre = new MultiLabel();
            if(!idPredictions.get(docID).isEmpty()) {
                String[] prediction = idPredictions.get(docID).split(",");


                for (int j = 0; j < prediction.length; j++) {

                    pre.addLabel(labelTranslator.toIntLabel(prediction[j].trim()));

                }

                predictions[i] = pre;
            }
        }


       return predictions;

    }


    public static MultiLabel reEncodeLabels(MultiLabel multiLabel, LabelTranslator labelTranslator, LabelTranslator newLabelTranslator){
        MultiLabel newMultiLabel = new MultiLabel();
        Set<Integer> internalLabels = multiLabel.getMatchedLabels();
        for( int i : internalLabels){
           String extLabel = labelTranslator.toExtLabel(i);
           newMultiLabel.addLabel(newLabelTranslator.toIntLabel(extLabel));

        }

        return newMultiLabel;
    }




















}
