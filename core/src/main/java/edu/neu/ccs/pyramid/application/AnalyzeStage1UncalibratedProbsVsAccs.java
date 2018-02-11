package edu.neu.ccs.pyramid.tmp;

import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class AnalyzeStage1UncalibratedProbsVsAccs {
    public static List<Pair<Double,Integer>> getUncalibratedProbs(){
        File file = new File("/mnt/home/zhenming/model_bak/radiology/IcdStage1_20171001/IcdStage1/reports_app3/test_99999999-9999-9999-9999-999999999999_reports/report.csv");
        List<String> lines = null;
        try {
            lines = FileUtils.readLines(file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        List<Pair<Double,Integer>> list = new ArrayList<>();
        for (String line: lines){
            String[] split = line.split("\t");
            if(split[2].equals("set")){
                double prob = Double.parseDouble(split[3]);
                int correct = Integer.parseInt(split[4]);
                Pair<Double, Integer> pair = new Pair<>(prob,correct);
                list.add(pair);

            }
        }
        return list;
    }

    public static List<Pair<Double,Integer>> getEachBucket(List<Pair<Double,Integer>> sortedList, int bucketIndex){
        List<Pair<Double,Integer>> bucketList = new ArrayList<>();
        for(int i=bucketIndex*1000; i<(bucketIndex+1)*1000;i++){
            bucketList.add(sortedList.get(i));
        }
        return bucketList;
    }

    public static String analyzeProbsVsAccs(List<Pair<Double,Integer>> list){

        int numBuckets = (int) Math.ceil(list.size()/1000.0);
        double[] correctNums = new double[numBuckets];
        double[] sumProbs = new double[numBuckets];
        double[] accuracy = new double[numBuckets];
        double[] average_confidence = new double[numBuckets];

        Comparator<Pair<Double,Integer>> probsComparator = Comparator.comparing(pair->pair.getFirst());

        List<Pair<Double,Integer>> sortedList = IntStream.range(0,list.size()).parallel().mapToObj(i->
             list.get(i)).sorted(probsComparator)
                .collect(Collectors.toList());

        for(int i=0;i<numBuckets;i++){
            List<Pair<Double,Integer>> bucketList = getEachBucket(sortedList,i);
            for(Pair<Double,Integer> pair : bucketList){
                sumProbs[i] += pair.getFirst();
                correctNums[i] += pair.getSecond();
            }
            if(i != numBuckets-1){
                accuracy[i] = correctNums[i]/1000.0;
                average_confidence[i] = sumProbs[i]/1000.0;
            }else{
                accuracy[i] = correctNums[i]/bucketList.size();
                average_confidence[i] = sumProbs[i]/bucketList.size();
            }
        }
        System.out.println("accs\t"+ Arrays.toString(accuracy));
        System.out.println("confidence\t"+Arrays.toString(average_confidence));
        DecimalFormat decimalFormat = new DecimalFormat("#0.0000");
        StringBuilder sb = new StringBuilder();
        sb.append("accuracy\t\t").append("average confidence\n");
        for (int i = 0; i < numBuckets; i++) {
            sb.append(decimalFormat.format(accuracy[i])).append("\t\t")
                    .append(decimalFormat.format(average_confidence[i])).append("\n");

        }

        String result = sb.toString();
        return result;

    }

    public static void main(String[] args) {
        String result = analyzeProbsVsAccs(getUncalibratedProbs());
        System.out.println(result);
    }










}
