package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.Displayer;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.CalibrationEval;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.eval.MLMeasures;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.*;
import edu.neu.ccs.pyramid.multilabel_classification.br.SupportPredictor;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.BMDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;

import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.ParallelFileWriter;
import edu.neu.ccs.pyramid.util.ParallelStringMapper;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.Serializable;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class BRCalibration {
    public static void main(Config config) throws Exception{
        System.out.println(config);
        if (config.getBoolean("calibrate")){
            calibrate(config);
        }

        if (config.getBoolean("test")){
            test(config);
        }
    }
    public static void main(String[] args) throws Exception {
        Config config = new Config(args[0]);
        main(config);

    }


    private static void calibrate(Config config) throws Exception{


        System.out.println("start training calibrator");
        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.trainData"), DataSetType.ML_CLF_SPARSE, true);
        //todo
        MultiLabelClfDataSet cal = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.validData"), DataSetType.ML_CLF_SPARSE, true);

        CBM cbm = (CBM) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model").toFile());

        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(train);
        LabelCalibrator labelCalibrator = new LabelCalibrator(cbm, cal);


        List<Pair<Integer, Integer>> implications = null;
        if (config.getBoolean("implication")) {
            System.out.println("find implications");
            implications = findImplications(train.getMultiLabels(), train.getNumClasses());
            System.out.println("done");
        }


        double[][][] pairPriors = null;
        if (config.getBoolean("pairPrior")) {
            System.out.println("computing pair priors");
            pairPriors = computePairPriors(train.getMultiLabels(), train.getNumClasses());
            System.out.println("done");
        }


        Map<MultiLabel, Double> setPrior = new HashMap<>();
        Map<Integer, Double> cardPrior = new HashMap<>();
        for (MultiLabel multiLabel : train.getMultiLabels()) {
            double setCount = setPrior.getOrDefault(multiLabel, 0.0);
            setPrior.put(multiLabel, setCount + 1.0 / train.getNumDataPoints());
            double cardCount = cardPrior.getOrDefault(multiLabel.getNumMatchedLabels(), 0.0);
            cardPrior.put(multiLabel.getNumMatchedLabels(), cardCount + 1.0 / train.getNumDataPoints());
        }

        List<Pair<Integer, Integer>> finalImplications = implications;
        double[][][] finalPairPriors = pairPriors;
        List<Instance> instances = IntStream.range(0, cal.getNumDataPoints()).parallel()
                .boxed().flatMap(i -> expand(config, cal, i, setPrior, cardPrior, cbm, labelCalibrator, finalPairPriors, finalImplications, support).stream())
                .collect(Collectors.toList());

        ClfDataSet clfDataSet = createClfData(instances, train.getLabelTranslator());

        VectorCardSetCalibrator vectorCardSetCalibratorProductCali = new VectorCardSetCalibrator(clfDataSet, 1, 3);

        Serialization.serialize(labelCalibrator,Paths.get(config.getString("output.dir"),"label_calibrator").toFile());
        Serialization.serialize(vectorCardSetCalibratorProductCali,Paths.get(config.getString("output.dir"),"card_iso_set_calibrator").toFile());
        Serialization.serialize(setPrior,Paths.get(config.getString("output.dir"),"set_priors").toFile());
        Serialization.serialize(cardPrior,Paths.get(config.getString("output.dir"),"card_priors").toFile());
        Serialization.serialize(implications,Paths.get(config.getString("output.dir"),"implications").toFile());
        Serialization.serialize(pairPriors,Paths.get(config.getString("output.dir"),"pair_priors").toFile());

        System.out.println("finish training calibrator");


    }

    private static void test(Config config) throws Exception{
        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("input.testData"), DataSetType.ML_CLF_SPARSE, true);
        CBM cbm = (CBM) Serialization.deserialize(Paths.get(config.getString("output.dir"),"model").toFile());
        LabelCalibrator labelCalibrator = (LabelCalibrator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"label_calibrator").toFile());
        VectorCardSetCalibrator vectorCardSetCalibrator = (VectorCardSetCalibrator) Serialization.deserialize(Paths.get(config.getString("output.dir"),"card_iso_set_calibrator").toFile());
        List<MultiLabel> support = (List<MultiLabel>) Serialization.deserialize(Paths.get(config.getString("output.dir"),"support").toFile());
        List<Pair<Integer, Integer>> implications = (List<Pair<Integer, Integer>>) Serialization.deserialize(Paths.get(config.getString("output.dir"),"implications").toFile());
        double[][][] pairPriors = (double[][][]) Serialization.deserialize(Paths.get(config.getString("output.dir"),"pair_priors").toFile());
        Map<MultiLabel, Double> setPrior = (Map<MultiLabel, Double>) Serialization.deserialize(Paths.get(config.getString("output.dir"),"set_priors").toFile());
        Map<Integer, Double> cardPrior = (Map<Integer, Double>) Serialization.deserialize(Paths.get(config.getString("output.dir"),"card_priors").toFile());


        BRSupportPrecictor brSupportPrecictor = new BRSupportPrecictor(cbm, support, labelCalibrator);
        System.out.println("test performance");
        System.out.println(new MLMeasures(brSupportPrecictor, test));

        if (true) {
            System.out.println("calibration performance on test set");

            List<Instance> predictions = IntStream.range(0, test.getNumDataPoints()).parallel()
                    .boxed().map(i -> predictedBySupport(config, test, i, setPrior, cardPrior, cbm, labelCalibrator, pairPriors, implications, support))
                    .collect(Collectors.toList());

            eval(predictions, vectorCardSetCalibrator);

        }


        boolean simpleCSV = true;
        if (simpleCSV){
            File testDataFile = new File(config.getString("input.testData"));
            File csv = Paths.get(config.getString("output.dir"),testDataFile.getName()+"_report_calibrated","report.csv").toFile();
            csv.getParentFile().mkdirs();
            List<Integer> list = IntStream.range(0,test.getNumDataPoints()).boxed().collect(Collectors.toList());
            ParallelStringMapper<Integer> mapper = (list1, i) -> simplePredictionAnalysisCalibrated(config, cbm, labelCalibrator, vectorCardSetCalibrator,
                    test, i, support, implications, pairPriors, cardPrior, setPrior);
            ParallelFileWriter.mapToString(mapper,list, csv,100  );
        }
    }


    public static String simplePredictionAnalysisCalibrated(Config config,
                                                             CBM cbm,
                                                             LabelCalibrator labelCalibrator,
                                                             VectorCardSetCalibrator vectorCardSetCalibrator,
                                                             MultiLabelClfDataSet dataSet,
                                                             int dataPointIndex,
                                                             List<MultiLabel> support,
                                                             List<Pair<Integer, Integer>> implications,
                                                             double[][][] pairPriors,
                                                             Map<Integer, Double> cardPriors,
                                                             Map<MultiLabel, Double> setPrior
    ){
        StringBuilder sb = new StringBuilder();
        MultiLabel trueLabels = dataSet.getMultiLabels()[dataPointIndex];
        String id = dataSet.getIdTranslator().toExtId(dataPointIndex);
        LabelTranslator labelTranslator = dataSet.getLabelTranslator();
        double[] classProbs = cbm.predictClassProbs(dataSet.getRow(dataPointIndex));
        double[] calibratedClassProbs = labelCalibrator.calibratedClassProbs(classProbs);

        MultiLabel predicted = SupportPredictor.predict(calibratedClassProbs,support);

        List<Integer> classes = new ArrayList<Integer>();
        for (int k = 0; k < dataSet.getNumClasses(); k++){
            if (dataSet.getMultiLabels()[dataPointIndex].matchClass(k)
                    ||predicted.matchClass(k)){
                classes.add(k);
            }
        }

        Comparator<Pair<Integer,Double>> comparator = Comparator.comparing(pair->pair.getSecond());
        List<Pair<Integer,Double>> list = classes.stream().map(l -> {
            if (l < cbm.getNumClasses()) {
                return new Pair<>(l, calibratedClassProbs[l]);
            } else {
                return new Pair<>(l, 0.0);
            }
        }).sorted(comparator.reversed()).collect(Collectors.toList());
        for (Pair<Integer,Double> pair: list){
            int label = pair.getFirst();
            double prob = pair.getSecond();
            int match = 0;
            if (trueLabels.matchClass(label)){
                match=1;
            }
            sb.append(id).append("\t").append(labelTranslator.toExtLabel(label)).append("\t")
                    .append("single").append("\t").append(prob)
                    .append("\t").append(match).append("\n");
        }

        Vector feature = feature(config,cbm.computeBM(dataSet.getRow(dataPointIndex),0.001),predicted,setPrior,
                cardPriors,calibratedClassProbs,pairPriors,implications,Optional.empty());
        double probability = vectorCardSetCalibrator.calibrate(feature);


        List<Integer> predictedList = predicted.getMatchedLabelsOrdered();
        sb.append(id).append("\t");
        for (int i=0;i<predictedList.size();i++){
            sb.append(labelTranslator.toExtLabel(predictedList.get(i)));
            if (i!=predictedList.size()-1){
                sb.append(",");
            }
        }
        sb.append("\t");
        int setMatch = 0;
        if (predicted.equals(trueLabels)){
            setMatch=1;
        }
        sb.append("set").append("\t").append(probability).append("\t").append(setMatch).append("\n");
        return sb.toString();
    }


    private static CaliRes eval(List<Instance> predictions, VectorCalibrator calibrator){
        double mse = CalibrationEval.mse(generateStream(predictions,calibrator));
        double ace = CalibrationEval.absoluteError(generateStream(predictions,calibrator),10);
        double sharpness = CalibrationEval.sharpness(generateStream(predictions,calibrator),10);
        System.out.println("mse="+mse);
        System.out.println("absolute calibration error="+ace);
        System.out.println("square calibration error="+CalibrationEval.squareError(generateStream(predictions,calibrator),10));
        System.out.println("sharpness="+sharpness);
        System.out.println("variance="+CalibrationEval.variance(generateStream(predictions,calibrator)));
        System.out.println(Displayer.displayCalibrationResult(generateStream(predictions,calibrator)));
        CaliRes caliRes = new CaliRes();
        caliRes.mse = mse;
        caliRes.ace= ace;
        caliRes.sharpness = sharpness;
        return caliRes;
    }

    private static List<Pair<Integer,Integer>> findImplications(MultiLabel[] multiLabels, int numClasses){
        List<Pair<Integer,Integer>> implications = new ArrayList<>();
        for (int l=0;l<numClasses;l++){
            for (int m=0;m<numClasses;m++){
                if (m!=l&&imply(l,m,multiLabels)){
                    System.out.println(l+" implies "+m);
                    implications.add(new Pair<>(l,m));
                }
            }
        }
        return implications;
    }

    private static boolean satisfy(MultiLabel multiLabel, List<Pair<Integer,Integer>> implications){
        for (Pair<Integer,Integer> implication: implications){
            if (!satisfy(multiLabel,implication)){
                return false;
            }
        }
        return true;
    }

    private static boolean satisfy(MultiLabel multiLabel, Pair<Integer,Integer> implication){
        if (multiLabel.matchClass(implication.getFirst())&&!multiLabel.matchClass(implication.getSecond())){
            return false;
        }
        return true;
    }


    private static boolean imply(int l, int m, MultiLabel[] multiLabels){
        for (MultiLabel multiLabel: multiLabels){
            if (multiLabel.matchClass(l)&&!multiLabel.matchClass(m)){
                return false;
            }
        }
        return true;
    }



    private static double priorF1(MultiLabel multiLabel, Map<MultiLabel,Double> setPriors){
        double sum = 0;
        for (Map.Entry<MultiLabel,Double> entry: setPriors.entrySet()){
            sum += entry.getValue()* FMeasure.f1(multiLabel, entry.getKey());
        }
        return sum;
    }

    private static double[][][] computePairPriors(MultiLabel[] multiLabels, int numClasses){
        double[][][] pairPriors = new double[numClasses][numClasses][4];
        IntStream.range(0,numClasses).parallel()
                .forEach(l->{
                    for (int j=l+1;j<numClasses;j++){
                        updatePairPrior(pairPriors,multiLabels,l,j);
                    }
                });
        return pairPriors;
    }

    private static void updatePairPrior(double[][][] pairPriors, MultiLabel[] multiLabels, int l, int j){
        for (MultiLabel multiLabel: multiLabels){
            if (multiLabel.matchClass(l)&&multiLabel.matchClass(j)){
                pairPriors[l][j][0] += 1.0/multiLabels.length;
            }
            if (multiLabel.matchClass(l)&&!multiLabel.matchClass(j)){
                pairPriors[l][j][1]  += 1.0/multiLabels.length;
            }
            if (!multiLabel.matchClass(l)&&multiLabel.matchClass(j)){
                pairPriors[l][j][2]  += 1.0/multiLabels.length;
            }
            if (!multiLabel.matchClass(l)&&!multiLabel.matchClass(j)){
                pairPriors[l][j][3]  += 1.0/multiLabels.length;
            }
        }
    }

    /**
     *
     * @param multiLabel
     * @param pairPriors [l][j][0] = p(both=1);[l][j][1] = p(only l=1);[l][j][2] = p(only j=1);[l][j][3] = p(both = 0)
     * @return
     */
    private static double pairCompatibility(MultiLabel multiLabel, double[][][] pairPriors){
        double min = Double.POSITIVE_INFINITY;
        int numClasses = pairPriors.length;
        for (int l=0;l<numClasses;l++){
            for (int j=l+1;j<numClasses;j++){
                double s=0;
                if (multiLabel.matchClass(l)&&multiLabel.matchClass(j)){
                    s = pairPriors[l][j][0];
                }
                if (multiLabel.matchClass(l)&&!multiLabel.matchClass(j)){
                    s = pairPriors[l][j][1];
                }
                if (!multiLabel.matchClass(l)&&multiLabel.matchClass(j)){
                    s = pairPriors[l][j][2];
                }
                if (!multiLabel.matchClass(l)&&!multiLabel.matchClass(j)){
                    s = pairPriors[l][j][3];
                }
                min = Math.min(min,s);
            }
        }
        return min;
    }


    private static Stream<Pair<Double,Integer>> generateStream(List<Instance> predictions, VectorCalibrator vectorCalibrator){
        return predictions.stream()
                .parallel().map(pred->new Pair<>(vectorCalibrator.calibrate(pred.vector),(int)pred.correctness));
    }





    private static ClfDataSet createClfData(List<Instance> instances, LabelTranslator labelTranslator){

        ClfDataSet clfDataSet = ClfDataSetBuilder.getBuilder()
                .numDataPoints(instances.size())
                .numFeatures(instances.get(0).vector.size())
                .numClasses(2)
                .dense(false)
                .build();
        for (int i=0;i<instances.size();i++){
            for (int j=0;j<clfDataSet.getNumFeatures();j++){
                clfDataSet.setFeatureValue(i,j,instances.get(i).vector.get(j));
            }
            clfDataSet.setLabel(i,(int)instances.get(i).correctness);
        }
        FeatureList featureList = new FeatureList();
        Feature feature0 = new Feature();
        feature0.setName("setPrior");
        Feature feature1 = new Feature();
        feature1.setName("br prob");
        Feature feature2 = new Feature();
        feature2.setName("card prior");
        Feature feature3 = new Feature();
        feature3.setName("card");
        Feature feature4 = new Feature();
        feature4.setName("pairPrior");
        Feature feature5 = new Feature();
        feature5.setName("f1 Prior");
        Feature feature6 = new Feature();
        feature6.setName("cbm prob");
        Feature feature7 = new Feature();
        feature7.setName("implication");
        Feature feature8 = new Feature();
        feature8.setName("position");
        featureList.add(feature0);
        featureList.add(feature1);
        featureList.add(feature2);
        featureList.add(feature3);
        featureList.add(feature4);
        featureList.add(feature5);
        featureList.add(feature6);
        featureList.add(feature7);
        featureList.add(feature8);
        for (int l=0;l<labelTranslator.getNumClasses();l++){
            Feature feature = new Feature();
            feature.setName("label_"+labelTranslator.toExtLabel(l));
            featureList.add(feature);
        }
        for (int l=0;l<labelTranslator.getNumClasses();l++){
            Feature feature = new Feature();
            feature.setName("label_"+labelTranslator.toExtLabel(l)+"_prob");
            featureList.add(feature);
        }
        clfDataSet.setFeatureList(featureList);
        return clfDataSet;
    }

    private static List<Instance> expand(Config config, MultiLabelClfDataSet dataSet, int index,
                                         Map<MultiLabel,Double> setPriors, Map<Integer,Double> cardPriors,
                                         CBM cbm, LabelCalibrator labelCalibrator,
                                         double[][][] pairPriors,List<Pair<Integer,Integer>> implications,
                                         List<MultiLabel> support){
        double[] marginals = labelCalibrator.calibratedClassProbs(cbm.predictClassProbs(dataSet.getRow(index)));
        MultiLabel prediction = SupportPredictor.predict(marginals, support);
        Map<MultiLabel, Integer> positionMap = positionMap(marginals);
        List<Instance> instances = new ArrayList<>();
        Set<MultiLabel> candidates = new HashSet<>();
        MultiLabel empty = new MultiLabel();
        candidates.add(prediction);
        candidates.add(empty);
        candidates.add(dataSet.getMultiLabels()[index]);
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        int top = config.getInt("calibrator.train.numCandidates");
        for (int i=0;i<top;i++){
            MultiLabel multiLabel = dynamicProgramming.nextHighestVector();
            candidates.add(multiLabel);
        }
        BMDistribution bmDistribution = cbm.computeBM(dataSet.getRow(index),0.001);
        for (MultiLabel multiLabel: candidates){
            instances.add(createInstance(config, bmDistribution, multiLabel,dataSet.getMultiLabels()[index],setPriors,cardPriors,marginals,pairPriors,implications,Optional.of(positionMap)));
        }
        return instances;
    }




    private static Instance predictedBySupport(Config config, MultiLabelClfDataSet dataSet, int index, Map<MultiLabel,Double> setPriors,
                                               Map<Integer,Double> cardPriors, CBM cbm, LabelCalibrator labelCalibrator,
                                               double[][][] pairPriors, List<Pair<Integer,Integer>> implications, List<MultiLabel> support){

        double[] marginals = labelCalibrator.calibratedClassProbs(cbm.predictClassProbs(dataSet.getRow(index)));
        MultiLabel prediction = SupportPredictor.predict(marginals, support);
        BMDistribution bmDistribution = cbm.computeBM(dataSet.getRow(index),0.001);
        return createInstance(config, bmDistribution,prediction,dataSet.getMultiLabels()[index],setPriors,cardPriors,marginals, pairPriors, implications, Optional.empty());
    }

    private static Instance createInstance(Config config, BMDistribution bmDistribution, MultiLabel multiLabel, MultiLabel groundtruth, Map<MultiLabel,Double> setPriors,
                                           Map<Integer,Double> cardPriors, double[] calibratedMarginals,
                                           double[][][] pairPriors, List<Pair<Integer,Integer>> implications,
                                           Optional<Map<MultiLabel,Integer>> positionMap){
        Instance instance = new Instance();
        instance.vector=feature(config, bmDistribution, multiLabel,setPriors,cardPriors,calibratedMarginals, pairPriors, implications, positionMap);
        instance.correctness = 0;
        if (multiLabel.equals(groundtruth)){
            instance.correctness=1;
        }
        return instance;
    }

    private static double truncatedLog(double score){
        if (score<1E-30){
            return Math.log(1E-30);
        }
        return Math.log(score);
    }

    private static Vector feature(Config config, BMDistribution bmDistribution, MultiLabel multiLabel, Map<MultiLabel,Double> setPriors,
                                  Map<Integer,Double> cardPriors, double[] calibratedMarginals,
                                  double[][][] pairPriors, List<Pair<Integer,Integer>> implications,
                                  Optional<Map<MultiLabel,Integer>> positionMap){
        boolean logScale = config.getBoolean("logScale");
        int numLabels = calibratedMarginals.length;
        Vector vector = new RandomAccessSparseVector(9+numLabels+numLabels);
        if (config.getBoolean("setPrior")){
            if (logScale){
                vector.set(0,truncatedLog(empiricalPrior(multiLabel, setPriors)));
            } else {
                vector.set(0,empiricalPrior(multiLabel, setPriors));
            }

        }
        if (config.getBoolean("brProb")){
            if (logScale){
                vector.set(1,truncatedLog(brProb(multiLabel,calibratedMarginals)));
            } else {
                vector.set(1,brProb(multiLabel,calibratedMarginals));
            }
        }

        if (config.getBoolean("cardPrior")){
            if (logScale){
                vector.set(2,truncatedLog(priorOfCard(multiLabel,cardPriors)));
            } else {
                vector.set(2,priorOfCard(multiLabel,cardPriors));
            }
        }
        if (config.getBoolean("card")){
            vector.set(3,multiLabel.getNumMatchedLabels());
        }
        if (config.getBoolean("pairPrior")){
            vector.set(4,pairCompatibility(multiLabel,pairPriors));
        }

        if (config.getBoolean("f1Prior")){
            vector.set(5,priorF1(multiLabel,setPriors));
        }

        if (config.getBoolean("cbmProb")){
            if (logScale){
                vector.set(6,bmDistribution.logProbability(multiLabel));
            } else {
                vector.set(6,Math.exp(bmDistribution.logProbability(multiLabel)));
            }

        }


        if (config.getBoolean("implication")){
            if (satisfy(multiLabel,implications)){
                vector.set(7,1);
            }
        }

        if (config.getBoolean("position")){
            int pos;
            if (positionMap.isPresent()){
                pos = positionMap.get().getOrDefault(multiLabel,Integer.MAX_VALUE);
            } else {
                pos = findPosition(multiLabel,calibratedMarginals);
            }
            vector.set(8,pos);
        }

        if (config.getBoolean("encodeLabel")){
            for (int l:multiLabel.getMatchedLabels()){
                //skip new labels
                if (l<numLabels){
                    vector.set(l+9,1);
                }
            }
        }


        if (config.getBoolean("labelProbs")){
            for (int l=0;l<numLabels;l++){
                if (multiLabel.matchClass(l)){
                    if (logScale){
                        vector.set(9+numLabels+l,truncatedLog(calibratedMarginals[l]));
                    } else {
                        vector.set(9+numLabels+l,calibratedMarginals[l]);
                    }

                } else {
                    if (logScale){
                        vector.set(9+numLabels+l,truncatedLog(1-calibratedMarginals[l]));
                    } else {
                        vector.set(9+numLabels+l,1-calibratedMarginals[l]);
                    }

                }
            }
        }

        return vector;
    }

    private static int findPosition(MultiLabel multiLabel, double[] marginals){
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        int upper = 1000;
        for (int i=0;i<upper;i++) {
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            if (candidate.equals(multiLabel)) {
                return i;
            }
        }
        return Integer.MAX_VALUE;
    }

    private static Map<MultiLabel, Integer> positionMap(double[] marginals){
        Map<MultiLabel, Integer> map = new HashMap<>();
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        int upper = 1000;
        for (int i=0;i<upper;i++) {
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            map.put(candidate,i);
        }
        return map;
    }

    private static double empiricalPrior(MultiLabel multiLabel, Map<MultiLabel,Double> priors){
        return priors.getOrDefault(multiLabel,0.0);
    }

    private static double brProb(MultiLabel multiLabel, double[] calibratedMarginals){
        double prod = 1;
        for (int l=0;l<calibratedMarginals.length;l++){
            if (multiLabel.matchClass(l)){
                prod *= calibratedMarginals[l];
            } else {
                prod *= 1-calibratedMarginals[l];
            }
        }
        return prod;
    }

    private static double priorOfCard(MultiLabel multiLabel, Map<Integer,Double> priors){
        return priors.getOrDefault(multiLabel.getNumMatchedLabels(),0.0);
    }

    private static class Instance{
        Vector vector;
        double correctness;
    }



    public static class CaliRes implements Serializable {
        public static final long serialVersionUID = 446782166720638575L;
        public double mse;
        public double ace;
        public double sharpness;
    }


    public static class BRSupportPrecictor implements MultiLabelClassifier{
        CBM cbm;
        List<MultiLabel> support;
        LabelCalibrator labelCalibrator;

        public BRSupportPrecictor(CBM cbm, List<MultiLabel> support, LabelCalibrator labelCalibrator) {
            this.cbm = cbm;
            this.support = support;
            this.labelCalibrator = labelCalibrator;
        }

        @Override
        public int getNumClasses() {
            return cbm.getNumClasses();
        }

        @Override
        public MultiLabel predict(Vector vector) {
            double[] marginals = cbm.predictClassProbs(vector);
            double[] calibratedMarginals = labelCalibrator.calibratedClassProbs(marginals);
            return SupportPredictor.predict(calibratedMarginals,support);
        }

        @Override
        public FeatureList getFeatureList() {
            return null;
        }

        @Override
        public LabelTranslator getLabelTranslator() {
            return null;
        }
    }

}
