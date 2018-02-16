package edu.neu.ccs.pyramid.regression;

import edu.neu.ccs.pyramid.multilabel_classification.imlgb.BucketInfo;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * follows the algorithm here:
 * http://stat.wikia.com/wiki/Isotonic_regression
 */

public class IsotonicRegression implements Serializable{
    private static final long serialVersionUID = 1L;
    //sorted locations
    private double[] locations;
    private double[] values;

    /**
     *
     * @param locations unsorted
     * @param numbers
     */
    public IsotonicRegression(double[] locations, double[] numbers) {
        List<Pair<Double, Double>> sorted = IntStream.range(0, locations.length).mapToObj(i->new Pair<>(locations[i], numbers[i]))
                .sorted(Comparator.comparing(pair->pair.getFirst())).collect(Collectors.toList());
        double[] sortedLocations  = sorted.stream().mapToDouble(p->p.getFirst()).toArray();
        double[] sortedNumbers  = sorted.stream().mapToDouble(p->p.getSecond()).toArray();
        double[] weights = new double[numbers.length];
        Arrays.fill(weights, 1.0);
        this.locations=sortedLocations;
        this.values = fit(sortedNumbers, weights);
    }

    /**
     *
     * @param locations unsorted
     * @param numbers
     */
    public IsotonicRegression(double[] locations, double[] numbers, double[] weights) {
        List<Element> sorted = IntStream.range(0, locations.length).mapToObj(i->new Element(locations[i], numbers[i], weights[i]))
                .sorted(Comparator.comparing(element->element.location)).collect(Collectors.toList());
        double[] sortedLocations  = sorted.stream().mapToDouble(p->p.location).toArray();
        double[] sortedNumbers  = sorted.stream().mapToDouble(p->p.number).toArray();
        double[] sortedWeights  = sorted.stream().mapToDouble(p->p.weight).toArray();
        this.locations=sortedLocations;
        this.values = fit(sortedNumbers, sortedWeights);
    }

    public IsotonicRegression(WeightedInput weightedInput) {
        this(weightedInput.locationNonEmpty, weightedInput.accsNonempty, weightedInput.countsNonEmpty);
    }



    public IsotonicRegression(Stream<Pair<Double,Integer>> stream){
        this(new WeightedInput(stream));
    }


    public String displayCalibrationResult(Stream<Pair<Double, Integer>> stream){
        final int numBuckets = 10;
        double bucketLength = 1.0/numBuckets;

        BucketInfo empty = new BucketInfo(numBuckets);
        BucketInfo total;
        total = stream.map(doubleIntegerPair -> {
            double probs = this.predict(doubleIntegerPair.getFirst());
            double[] sum = new double[numBuckets];
            double[] sumProbs = new double[numBuckets];
            double[] count = new double[numBuckets];
            int index = (int)Math.floor(probs/bucketLength);
            if (index<0){
                index=0;
            }
            if (index>=numBuckets){
                index = numBuckets-1;
            }
            count[index] += 1;
            sumProbs[index] += probs;
            sum[index]+=doubleIntegerPair.getSecond();
            return new BucketInfo(count, sum,sumProbs);
        }).reduce(empty, BucketInfo::add, BucketInfo::add);
        double[] counts = total.getCounts();
        double[] correct = total.getSums();
        double[] sumProbs = total.getSumProbs();
        double[] accs = new double[counts.length];
        double[] average_confidence = new double[counts.length];

        for (int i = 0; i < counts.length; i++) {
            accs[i] = correct[i] / counts[i];
        }
        for (int j = 0; j < counts.length; j++) {
            average_confidence[j] = sumProbs[j] / counts[j];
        }

        DecimalFormat decimalFormat = new DecimalFormat("#0.0000");
        StringBuilder sb = new StringBuilder();
        sb.append("interval\t\t").append("total\t\t").append("correct\t\t").append("incorrect\t\t").append("accuracy\t\t").append("average confidence\n");
        for (int i = 0; i < 10; i++) {
            sb.append("[").append(decimalFormat.format(i * 0.1)).append(",")
                    .append(decimalFormat.format((i + 1) * 0.1)).append("]")
                    .append("\t\t").append(counts[i]).append("\t\t").append(correct[i]).append("\t\t")
                    .append(counts[i] - correct[i]).append("\t\t").append(decimalFormat.format(accs[i])).append("\t\t")
                    .append(decimalFormat.format(average_confidence[i])).append("\n");

        }

        String result = sb.toString();
        return result;

    }




    public double[] getLocations() {
        return locations;
    }

    public double[] getValues() {
        return values;
    }

    public double predict(double location){
        if (location<=locations[0]){
            return values[0];
        }

        if (location>=locations[locations.length-1]){
            return values[values.length-1];
        }

        //use binary search
        int l = locations.length;
        int start=0;
        int end = l-1;

        while(end-start>1){
            int middle = (start+end)/2;
            if (locations[middle]==location){
                return values[middle];
            } else if (locations[middle]>location){
                end = middle;
            } else{
                start = middle;
            }
        }

        if (start==end){
            return values[start];
        } else {
            return (location-locations[start])*(values[end]-values[start])/(locations[end]-locations[start])+values[start];
        }

    }



    /**
     *
     * @param numbers sorted by locations
     */
    public IsotonicRegression(double[] numbers) {
        double[] weights = new double[numbers.length];
        Arrays.fill(weights, 1.0);
        this.values = fit(numbers, weights);
    }

    private double[] fit1Based(double[] a, double[] w){
        double[] aprime = new double[a.length];
        double[] wprime = new double[w.length];
        int[] s = new int[a.length];
        aprime[1]= a[1];
        wprime[1] = w[1];
        int j=1;
        s[0]=0;
        s[1]=1;
        for (int i=2;i<a.length;i++){
            j += 1;
            aprime[j] = a[i];
            wprime[j] = w[i];
            while (j>1 && aprime[j]<aprime[j-1]){
                aprime[j-1] = (wprime[j]*aprime[j]+wprime[j-1]*aprime[j-1])/(wprime[j]+wprime[j-1]);
                wprime[j-1] = wprime[j]+wprime[j-1];
                j -=1;
            }
            s[j] = i;
        }
        double[] parameters = new double[a.length];
        for (int k=1;k<=j;k++){
            for (int l=s[k-1]+1;l<=s[k];l++){
                parameters[l] = aprime[k];
            }
        }
        return parameters;
    }

    private double[] fit(double[] a, double[] w){
        double[] a1Based = new double[a.length+1];
        double[] w1Based = new double[w.length+1];
        for (int i=0;i<a.length;i++){
            a1Based[i+1]=a[i];
        }
        for (int i=0;i<w.length;i++){
            w1Based[i+1]=w[i];
        }
        double[] p =  fit1Based(a1Based, w1Based);
        return Arrays.copyOfRange(p,1,p.length);
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("IsotonicRegression{");
        sb.append("locations=").append(Arrays.toString(locations));
        sb.append(", values=").append(Arrays.toString(values));
        sb.append('}');
        return sb.toString();
    }

    private static class Element{
        Element(double location, double number, double weight) {
            this.location = location;
            this.number = number;
            this.weight = weight;
        }

        private double location;
        private double number;
        private double weight;
    }



    private static class WeightedInput{
        private double[] locationNonEmpty;
        private double[] accsNonempty;
        private double[] countsNonEmpty;

        public WeightedInput(Stream<Pair<Double,Integer>> stream) {
            final int numBuckets = 10000;
            double bucketLength = 1.0/numBuckets;
            double[] locations = new double[numBuckets];
            for (int i=0;i<numBuckets;i++){
                locations[i]= i*bucketLength + 0.5*bucketLength;
            }

            BucketInfo empty = new BucketInfo(numBuckets);
            BucketInfo total;
            total = stream.map(doubleIntegerPair -> {
                double probs =  doubleIntegerPair.getFirst();
                double[] sum = new double[numBuckets];
                double[] sumProbs = new double[numBuckets];
                double[] count = new double[numBuckets];
                int index = (int)Math.floor(probs/bucketLength);
                if (index<0){
                    index=0;
                }
                if (index>=numBuckets){
                    index = numBuckets-1;
                }
                count[index] += 1;
                sumProbs[index] += probs;
                sum[index]+=doubleIntegerPair.getSecond();
                return new BucketInfo(count, sum,sumProbs);
            }).reduce(empty, BucketInfo::add, BucketInfo::add);
            double[] counts = total.counts;
            double[] sums = total.sums;


            this.countsNonEmpty = IntStream.range(0,counts.length).filter(i->counts[i]!=0)
                    .mapToDouble(i->counts[i]).toArray();
            this.locationNonEmpty = IntStream.range(0,counts.length).filter(i->counts[i]!=0)
                    .mapToDouble(i->locations[i]).toArray();
            double[] sumNonempty = IntStream.range(0,counts.length).filter(i->counts[i]!=0)
                    .mapToDouble(i->sums[i]).toArray();
            this. accsNonempty = IntStream.range(0,countsNonEmpty.length)
                    .mapToDouble(i->sumNonempty[i]/countsNonEmpty[i]).toArray();
        }
    }
}
