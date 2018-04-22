package edu.neu.ccs.pyramid.calibration;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.multilabel_classification.imlgb.*;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.*;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class StreamGenerator {

    private Config config;
    private IMLGradientBoosting boosting;
    private MultiLabelClfDataSet dataSet;
    private IMLGBLabelIsotonicScaling labelCali;

    public StreamGenerator(Config config, IMLGradientBoosting boosting,
                           MultiLabelClfDataSet dataSet, IMLGBLabelIsotonicScaling labelCali) {
        this.config = config;
        this.boosting = boosting;
        this.dataSet = dataSet;
        this.labelCali = labelCali;
    }

    public  Stream<Pair<Vector,Integer>> generateStream(){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .boxed().flatMap(this::generateStream);
    }

    public Stream<Pair<Double,Integer>> generateCalibratedStream(Regressor regressor){
        return generateStream().map(a->new Pair<>(regressor.predict(a.getFirst()),a.getSecond()));
    }

    public List<Double> calibratedProbsForSupport(Regressor regressor, Vector instance){
        double[] uncalibratedLabelProbs = boosting.predictClassProbs(instance);
        double[] calibratedLabelProbs = labelCali.calibratedClassProbs(uncalibratedLabelProbs);
        List<MultiLabel> candidates = boosting.getAssignments();
        double[] uncalibrated = new double[candidates.size()];
        if (config.getString("B").equals("1")){
            for (int i=0;i<candidates.size();i++){
                uncalibrated[i] = proba(candidates.get(i),uncalibratedLabelProbs);
            }
        } else {
            for (int i=0;i<candidates.size();i++){
                uncalibrated[i] = proba(candidates.get(i),calibratedLabelProbs);
            }
        }

        if (config.getString("C").equals("1")){
            double sum = MathUtil.arraySum(uncalibrated);
            for (int i=0;i<uncalibrated.length;i++){
                uncalibrated[i] /= sum;
            }
        }

        List<Double> calibrated = new ArrayList<>();
        for (int a=0;a<candidates.size();a++){
            Vector vector = new DenseVector(4);
            //todo add more dim
            vector.set(0, uncalibrated[a]);
            vector.set(1,candidates.get(a).getNumMatchedLabels());
            calibrated.add(regressor.predict(vector));
        }
        return calibrated;
    }

    public double calibratedSum(Regressor regressor){
        return IntStream.range(0, dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->calibratedSum(regressor,i)).average().getAsDouble();
    }



    public double calibratedSum(Regressor regressor, int data){
        double[] uncalibratedLabelProbs = boosting.predictClassProbs(dataSet.getRow(data));
        double[] calibratedLabelProbs = labelCali.calibratedClassProbs(uncalibratedLabelProbs);

        MultiLabel accPred = new SubsetAccPredictor(boosting).predict(dataSet.getRow(data));
        MultiLabel hammingPred = new HammingPredictor(boosting).predict(dataSet.getRow(data));
        MultiLabel accPredCalibrated = new CalibratedSetAccPredictor(boosting, labelCali).predict(dataSet.getRow(data));
        MultiLabel hammingPredCalibrated = new CalibratedHammingPredictor(boosting, labelCali).predict(dataSet.getRow(data));
        MultiLabel top = null;
        if (config.getString("A").equals("1") && config.getString("G").equals("1")){
            top = hammingPred;
        }

        if (config.getString("A").equals("1") && config.getString("G").equals("2")){
            top = hammingPredCalibrated;
        }


        if (config.getString("A").equals("2") && config.getString("G").equals("1")){
            top = accPred;
        }

        if (config.getString("A").equals("2") && config.getString("G").equals("2")){
            top = accPredCalibrated;
        }

        Set<MultiLabel> candidateSet = new HashSet<>();
        candidateSet.add(top);

        //todo always add hamming?
        candidateSet.add(hammingPred);
        candidateSet.add(hammingPredCalibrated);
        //todo empty set
        candidateSet.add(new MultiLabel());
        candidateSet.addAll(boosting.getAssignments());

        List<MultiLabel> candidates = new ArrayList<>(candidateSet);

        double[] probabilities = new double[candidates.size()];
        if (config.getString("B").equals("1")){
            for (int i=0;i<candidates.size();i++){
                probabilities[i] = proba(candidates.get(i),uncalibratedLabelProbs);
            }
        } else {
            for (int i=0;i<candidates.size();i++){
                probabilities[i] = proba(candidates.get(i),calibratedLabelProbs);
            }
        }

        if (config.getString("C").equals("1")){
            double sum = MathUtil.arraySum(probabilities);
            for (int i=0;i<probabilities.length;i++){
                probabilities[i] /= sum;
            }
        }

        List<MultiLabel> fcandidates = candidates;

        Stream<Item> stream = IntStream.range(0,fcandidates.size())
                .mapToObj(i->{
                    Vector vector = new DenseVector(4);
                    //todo add more dim
                    vector.set(0, probabilities[i]);
                    vector.set(1,fcandidates.get(i).getNumMatchedLabels());
                    Item item = new Item();
                    item.multiLabel = fcandidates.get(i);
                    if (fcandidates.get(i).equals(dataSet.getMultiLabels()[data])){
                        item.label=1;
                    }
                    item.vector = vector;
                    return item;
                });

        return stream.mapToDouble(item->regressor.predict(item.vector)).sum();
    }

    private Stream<Pair<Vector,Integer>> generateStream(int data){
        double[] uncalibratedLabelProbs = boosting.predictClassProbs(dataSet.getRow(data));
        double[] calibratedLabelProbs = labelCali.calibratedClassProbs(uncalibratedLabelProbs);

        MultiLabel accPred = new SubsetAccPredictor(boosting).predict(dataSet.getRow(data));
        MultiLabel hammingPred = new HammingPredictor(boosting).predict(dataSet.getRow(data));
        MultiLabel accPredCalibrated = new CalibratedSetAccPredictor(boosting, labelCali).predict(dataSet.getRow(data));
        MultiLabel hammingPredCalibrated = new CalibratedHammingPredictor(boosting, labelCali).predict(dataSet.getRow(data));
        MultiLabel top = null;
        if (config.getString("A").equals("1") && config.getString("G").equals("1")){
            top = hammingPred;
        }

        if (config.getString("A").equals("1") && config.getString("G").equals("2")){
            top = hammingPredCalibrated;
        }


        if (config.getString("A").equals("2") && config.getString("G").equals("1")){
            top = accPred;
        }

        if (config.getString("A").equals("2") && config.getString("G").equals("2")){
            top = accPredCalibrated;
        }

        Set<MultiLabel> candidateSet = new HashSet<>();
        candidateSet.add(top);

        //todo always add hamming?
        candidateSet.add(hammingPred);
        candidateSet.add(hammingPredCalibrated);
        //todo empty set
        candidateSet.add(new MultiLabel());
        candidateSet.addAll(boosting.getAssignments());

        List<MultiLabel> candidates = new ArrayList<>(candidateSet);

        double[] probabilities = new double[candidates.size()];
        if (config.getString("B").equals("1")){
            for (int i=0;i<candidates.size();i++){
                probabilities[i] = proba(candidates.get(i),uncalibratedLabelProbs);
            }
        } else {
            for (int i=0;i<candidates.size();i++){
                probabilities[i] = proba(candidates.get(i),calibratedLabelProbs);
            }
        }

        if (config.getString("C").equals("1")){
            double sum = MathUtil.arraySum(probabilities);

            if (sum==0){
                System.out.println("error sum=0");
                System.out.println(Arrays.toString(calibratedLabelProbs));
                System.out.println("hamm prob "+proba(hammingPred,calibratedLabelProbs));
            }
            for (int i=0;i<probabilities.length;i++){
                probabilities[i] /= sum;
            }
        }

        List<MultiLabel> fcandidates = candidates;

        Stream<Item> stream = IntStream.range(0,fcandidates.size())
                .mapToObj(i->{
                    Vector vector = new DenseVector(4);
                    //todo add more dim
                    vector.set(0, probabilities[i]);
                    if (!Double.isFinite(probabilities[i])){
                        System.out.println("error: prob");
                    }
                    vector.set(1,fcandidates.get(i).getNumMatchedLabels());
                    Item item = new Item();
                    item.multiLabel = fcandidates.get(i);
                    if (fcandidates.get(i).equals(dataSet.getMultiLabels()[data])){
                        item.label=1;
                    }
                    item.vector = vector;
                    return item;
                });

        MultiLabel fTop = top;
        if (config.getString("D").equals("2")){
            return stream.map(item->new Pair<>(item.vector,item.label));
        } else {
            return stream.filter(item->item.multiLabel.equals(fTop)).map(item->new Pair<>(item.vector,item.label));
        }

    }

    private static double proba(MultiLabel multiLabel, double[] marginals){
        double product = 1;
        for (int l=0;l<marginals.length;l++){

            if (multiLabel.matchClass(l)){
                product *= marginals[l];
            } else {
                product *= 1-marginals[l];
            }
        }
        return product;
    }

    private static class Item{
        MultiLabel multiLabel;
        Vector vector;
        int label;
    }
}
