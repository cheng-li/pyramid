package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.util.Pair;

import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CTFT {


        public static CTFT.Summary findThreshold(Stream<Pair<Double, Double>> stream, double targetF1){
        Comparator<Pair<Double, Double>> comparator = Comparator.comparing(Pair::getFirst);
        List<Pair<Double,Double>> list = stream.sorted(comparator.reversed()).collect(Collectors.toList());
        CTFT.Summary summary = new CTFT.Summary();
        double sumF1 = 0;
        int size = list.size();
        int numCorrect = 0;

        for (int i = 0; i < size; i++){
            sumF1 += list.get(i).getSecond();
            if(list.get(i).getSecond() == 1.0){
                numCorrect += 1;
            }
            double f1 = (sumF1*1.0)/(i+1);
            double acc = (numCorrect*1.0)/(i+1);
            if (i==size-1||(i<size-1&&(!list.get(i).getFirst().equals(list.get(i+1).getFirst())))){
                if (f1 >= targetF1 ){
                    summary.confidenceThreshold = list.get(i).getFirst();
                    summary.autoCodingPercentage = (i+1)/(size*1.0);
                    summary.autoCodingF1 = f1;
                    summary.autoCodingAccuracy = acc;
                    summary.numAutoCoded = (i+1);
                    summary.numCorrectAutoCoded = numCorrect;
                }
            }

        }
        return summary;
    }

    public static CTFT.Summary applyThreshold(Stream<Pair<Double, Double>> stream, double confidenceThreshold){
        List<Pair<Double,Double>> list = stream.collect(Collectors.toList());
        int sum = 0;
        int correct = 0;
        double sumF1 = 0;
        int size = list.size();
        for (int i = 0; i<size; i++){
            if (list.get(i).getFirst() >= confidenceThreshold){
                sum++;
                sumF1 += list.get(i).getSecond();
                if(list.get(i).getSecond() == 1.0){
                    correct += 1;
                }
            }
        }
        CTFT.Summary summary = new CTFT.Summary();
        summary.confidenceThreshold=confidenceThreshold;
        summary.autoCodingPercentage = (sum*1.0)/size;
        summary.autoCodingAccuracy = (correct*1.0)/sum;
        summary.autoCodingF1 = (sumF1*1.0)/sum;
        summary.numAutoCoded = sum;
        summary.numCorrectAutoCoded = correct;
        return summary;
    }







    public static class Summary {
        double confidenceThreshold=1.1;
        double autoCodingPercentage =0;
        double autoCodingF1 =Double.NaN;
        double autoCodingAccuracy = Double.NaN;
        int numAutoCoded =0;
        int numCorrectAutoCoded =0;



        public double getConfidenceThreshold() {
            return confidenceThreshold;
        }

        public double getAutoCodingPercentage() {
            return autoCodingPercentage;
        }

        public double getAutoCodingF1() {
            return autoCodingF1;
        }

        public double getAutoCodingAccuracy(){ return autoCodingAccuracy;}

        public int getNumAutoCoded() {
            return numAutoCoded;
        }

        public int getNumCorrectAutoCoded(){ return numCorrectAutoCoded;}




        @Override
        public String toString() {
            return "Summary{" +
                    "confidenceThreshold=" + confidenceThreshold +
                    ", autoCodingPercentage=" + autoCodingPercentage +
                    ", autoCodingF1=" + autoCodingF1 +
                    ", autoCodingAccuracy=" + autoCodingAccuracy+
                    ", numAutoCoded=" + numAutoCoded +
                    ",numCorrectCoded="+ numCorrectAutoCoded+
                    '}';
        }
    }








}
