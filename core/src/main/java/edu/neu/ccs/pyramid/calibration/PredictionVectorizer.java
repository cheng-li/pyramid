package edu.neu.ccs.pyramid.calibration;


import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.FMeasure;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.BMDistribution;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.util.Pair;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PredictionVectorizer implements Serializable {
    private static final long serialVersionUID = 2L;
    private boolean logScale;
    private boolean setPrior;
    private boolean brProb;
    private boolean cardPrior;
    private boolean card;
    private boolean pairPrior;
    private boolean f1Prior;
    private boolean cbmProb;
    private boolean implication;
    private boolean position;
    private boolean encodeLabel;
    private boolean labelProbs;
    private boolean hierarchy;
    private Map<MultiLabel,Double> setPriors;
    private Map<Integer,Double> cardPriors;
    private double[][][] pairPriors;
    private List<Pair<Integer,Integer>> implications;
    private LabelCalibrator labelCalibrator;
    private int numCandidates;
    private Hierarchy hierarchyRelation;
    private String weight;

    private PredictionVectorizer(Builder builder) {
        logScale = builder.logScale;
        setPrior = builder.setPrior;
        brProb = builder.brProb;
        cardPrior = builder.cardPrior;
        card = builder.card;
        pairPrior = builder.pairPrior;
        f1Prior = builder.f1Prior;
        cbmProb = builder.cbmProb;
        implication = builder.implication;
        position = builder.position;
        encodeLabel = builder.encodeLabel;
        labelProbs = builder.labelProbs;
        numCandidates = builder.numCandidates;
        hierarchy = builder.hierarchy;
        weight = builder.weight;
    }



    public static Builder newBuilder() {
        return new Builder();
    }

    public LabelCalibrator getLabelCalibrator() {
        return labelCalibrator;
    }

    public void setHierarchyRelation(Hierarchy hierarchyRelation) {
        this.hierarchyRelation = hierarchyRelation;
    }

    public Vector feature(BMDistribution bmDistribution, MultiLabel multiLabel, double[] calibratedMarginals,
                          Optional<Map<MultiLabel,Integer>> positionMap){

        int numLabels = calibratedMarginals.length;
        Vector vector = new RandomAccessSparseVector(10+numLabels+numLabels);
        if (setPrior){
            if (logScale){
                vector.set(0,truncatedLog(empiricalPrior(multiLabel, setPriors)));
            } else {
                vector.set(0,empiricalPrior(multiLabel, setPriors));
            }

        }
        if (brProb){
            if (logScale){
                vector.set(1,truncatedLog(brProb(multiLabel,calibratedMarginals)));
            } else {
                vector.set(1,brProb(multiLabel,calibratedMarginals));
            }
        }

        if (cardPrior){
            if (logScale){
                vector.set(2,truncatedLog(priorOfCard(multiLabel,cardPriors)));
            } else {
                vector.set(2,priorOfCard(multiLabel,cardPriors));
            }
        }
        if (card){
            vector.set(3,multiLabel.getNumMatchedLabels());
        }
        if (pairPrior){
            vector.set(4,pairCompatibility(multiLabel,pairPriors));
        }

        if (f1Prior){
            vector.set(5,priorF1(multiLabel,setPriors));
        }

        if (cbmProb){
            if (logScale){
                vector.set(6,bmDistribution.logProbability(multiLabel));
            } else {
                vector.set(6,Math.exp(bmDistribution.logProbability(multiLabel)));
            }

        }


        if (implication){
            if (satisfy(multiLabel,implications)){
                vector.set(7,1);
            }
        }

        if (position){
            int pos;
            if (positionMap.isPresent()){
                pos = positionMap.get().getOrDefault(multiLabel,Integer.MAX_VALUE);
            } else {
                pos = findPosition(multiLabel,calibratedMarginals);
            }
            vector.set(8,pos);
        }

        if (hierarchy){
            if (hierarchyRelation.satisfy(multiLabel)){
                vector.set(9, 1);
            } else {
                vector.set(9, 0);
            }

        }

        if (encodeLabel){
            for (int l:multiLabel.getMatchedLabels()){
                //skip new labels
                if (l<numLabels){
                    vector.set(l+10,1);
                }
            }
        }


        if (labelProbs){
            for (int l=0;l<numLabels;l++){
                if (multiLabel.matchClass(l)){
                    if (logScale){
                        vector.set(10+numLabels+l,truncatedLog(calibratedMarginals[l]));
                    } else {
                        vector.set(10+numLabels+l,calibratedMarginals[l]);
                    }

                } else {
                    if (logScale){
                        vector.set(10+numLabels+l,truncatedLog(1-calibratedMarginals[l]));
                    } else {
                        vector.set(10+numLabels+l,1-calibratedMarginals[l]);
                    }

                }
            }
        }

        return vector;
    }


    public int[][] getMonotonicityConstraints(int numLabelsInModel){
        int[][] monotonicity = new int[1][10+numLabelsInModel+numLabelsInModel];

        monotonicity[0][0]=1;
        monotonicity[0][1]=1;
        monotonicity[0][2]=1;
        monotonicity[0][5]=1;
        monotonicity[0][6]=1;
        monotonicity[0][7]=1;
        monotonicity[0][8]=-1;
        monotonicity[0][9]=1;
        for (int l=0;l<numLabelsInModel;l++){
            monotonicity[0][10+numLabelsInModel+l]=1;
        }
        return monotonicity;
    }

    private Pair<RegDataSet,double[]> createData(List<Instance> instances, LabelTranslator trainLabelTranslator){
        RegDataSet regDataSet = RegDataSetBuilder.getBuilder()
                .numDataPoints(instances.size())
                .numFeatures(instances.get(0).vector.size())
                .dense(false)
                .build();
        double[] weights = new double[instances.size()];
        for (int i=0;i<instances.size();i++){
            for (int j=0;j<regDataSet.getNumFeatures();j++){
                regDataSet.setFeatureValue(i,j,instances.get(i).vector.get(j));
            }
            regDataSet.setLabel(i,instances.get(i).correctness);
            weights[i] = instances.get(i).weight;
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
        Feature feature9 = new Feature();
        feature9.setName("hierarchy");
        featureList.add(feature0);
        featureList.add(feature1);
        featureList.add(feature2);
        featureList.add(feature3);
        featureList.add(feature4);
        featureList.add(feature5);
        featureList.add(feature6);
        featureList.add(feature7);
        featureList.add(feature8);
        featureList.add(feature9);
        for (int l=0;l<trainLabelTranslator.getNumClasses();l++){
            Feature feature = new Feature();
            feature.setName("label_"+trainLabelTranslator.toExtLabel(l));
            featureList.add(feature);
        }

        for (int l=0;l<trainLabelTranslator.getNumClasses();l++){
            Feature feature = new Feature();
            feature.setName("label_"+trainLabelTranslator.toExtLabel(l)+"_prob");
            featureList.add(feature);
        }
        regDataSet.setFeatureList(featureList);
        return new Pair<>(regDataSet,weights);
    }

    public Pair<RegDataSet,double[]> createCaliTrainingData(MultiLabelClfDataSet calDataSet, CBM cbm){
        List<Instance> instances = IntStream.range(0, calDataSet.getNumDataPoints()).parallel()
                .boxed().flatMap(i -> expand(calDataSet.getRow(i),calDataSet.getMultiLabels()[i], cbm).stream())
                .collect(Collectors.toList());
        return createData(instances, cbm.getLabelTranslator());
    }

    private List<Instance> expand(Vector x, MultiLabel groundTruth,
                                         CBM cbm){
        double[] marginals = labelCalibrator.calibratedClassProbs(cbm.predictClassProbs(x));
        Map<MultiLabel, Integer> positionMap = positionMap(marginals);
        List<Instance> instances = new ArrayList<>();

        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        BMDistribution bmDistribution = cbm.computeBM(x,0.001);
        for (int i=0;i<numCandidates;i++){
            MultiLabel multiLabel = dynamicProgramming.nextHighestVector();
            Instance instance = createInstance(bmDistribution, multiLabel,groundTruth,marginals,Optional.of(positionMap));
            if (weight.equals("uniform")){
                instance.weight=1;
            } else if (weight.startsWith("expBase")){
                //expBase
                double base = Double.parseDouble(weight.substring(7));
                instance.weight = Math.pow(base,-1*i);
            } else if (weight.startsWith("downSampleNegRate")){
                double rate = Double.parseDouble(weight.substring(17));
                if (instance.correctness==1){
                    instance.weight=1;
                } else {
                    instance.weight=rate;
                }
            } else {
                throw new RuntimeException("unknown weight method");
            }
            instances.add(instance);
        }

        return instances;
    }

    public Instance createInstance(CBM cbm, Vector x, MultiLabel prediction, MultiLabel groundTruth){
        BMDistribution bmDistribution = cbm.computeBM(x,0.001);
        double[] uncali = cbm.predictClassProbs(x);
        double[] cali = labelCalibrator.calibratedClassProbs(uncali);
        return createInstance(bmDistribution, prediction, groundTruth,cali, Optional.empty());
    }

    public Vector feature(CBM cbm, Vector x, MultiLabel prediction){
        BMDistribution bmDistribution = cbm.computeBM(x,0.001);
        double[] uncali = cbm.predictClassProbs(x);
        double[] cali = labelCalibrator.calibratedClassProbs(uncali);
        return feature(bmDistribution, prediction,cali, Optional.empty());
    }

    private Instance createInstance(BMDistribution bmDistribution, MultiLabel multiLabel, MultiLabel groundtruth, double[] calibratedMarginals,
                                                         Optional<Map<MultiLabel,Integer>> positionMap){
        Instance instance = new Instance();
        instance.vector=feature(bmDistribution, multiLabel,calibratedMarginals, positionMap);
        instance.correctness = 0;
        if (multiLabel.equals(groundtruth)){
            instance.correctness=1;
        }
        return instance;
    }


    public static class Instance{
        public Vector vector;
        public double correctness;
        public double weight=1;
    }



    private static double truncatedLog(double score){
        if (score<1E-30){
            return Math.log(1E-30);
        }
        return Math.log(score);
    }



    private int findPosition(MultiLabel multiLabel, double[] marginals){
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        for (int i=0;i<numCandidates;i++) {
            MultiLabel candidate = dynamicProgramming.nextHighestVector();
            if (candidate.equals(multiLabel)) {
                return i;
            }
        }
        return Integer.MAX_VALUE;
    }

    public Map<MultiLabel, Integer> positionMap(double[] marginals){
        Map<MultiLabel, Integer> map = new HashMap<>();
        DynamicProgramming dynamicProgramming = new DynamicProgramming(marginals);
        for (int i=0;i<numCandidates;i++) {
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

    private static List<Pair<Integer,Integer>> findImplications(MultiLabel[] multiLabels, int numClasses){
        List<Pair<Integer,Integer>> implications = new ArrayList<>();
        for (int l=0;l<numClasses;l++){
            for (int m=0;m<numClasses;m++){
                if (m!=l&&imply(l,m,multiLabels)){
//                    logger.info(l+" implies "+m);
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


    public static final class Builder {
        private boolean logScale=false;
        private boolean setPrior=true;
        private boolean brProb=true;
        private boolean cardPrior=true;
        private boolean card=true;
        private boolean pairPrior=false;
        private boolean f1Prior=false;
        private boolean cbmProb=true;
        private boolean implication=false;
        private boolean position=true;
        private boolean encodeLabel=true;
        private boolean labelProbs=false;
        private int numCandidates=50;
        private boolean hierarchy=false;
        private String weight="uniform";

        private Builder() {
        }

        public Builder logScale(boolean val) {
            logScale = val;
            return this;
        }

        public Builder setPrior(boolean val) {
            setPrior = val;
            return this;
        }

        public Builder brProb(boolean val) {
            brProb = val;
            return this;
        }

        public Builder cardPrior(boolean val) {
            cardPrior = val;
            return this;
        }

        public Builder card(boolean val) {
            card = val;
            return this;
        }

        public Builder pairPrior(boolean val) {
            pairPrior = val;
            return this;
        }

        public Builder f1Prior(boolean val) {
            f1Prior = val;
            return this;
        }

        public Builder cbmProb(boolean val) {
            cbmProb = val;
            return this;
        }

        public Builder implication(boolean val) {
            implication = val;
            return this;
        }

        public Builder position(boolean val) {
            position = val;
            return this;
        }

        public Builder encodeLabel(boolean val) {
            encodeLabel = val;
            return this;
        }

        public Builder labelProbs(boolean val) {
            labelProbs = val;
            return this;
        }

        public Builder numCandidates(int val) {
            numCandidates = val;
            return this;
        }

        public Builder hierarchy(boolean val){
            hierarchy = val;
            return this;
        }

        public Builder weight(String val){
            weight = val;
            return this;
        }



        public PredictionVectorizer build(MultiLabelClfDataSet dataSet, LabelCalibrator labelCalibrator) {
            PredictionVectorizer predictionVectorizer = new PredictionVectorizer(this);
            if (implication) {
                predictionVectorizer.implications = findImplications(dataSet.getMultiLabels(), dataSet.getNumClasses());
            }

            if (pairPrior) {
                predictionVectorizer.pairPriors = computePairPriors(dataSet.getMultiLabels(), dataSet.getNumClasses());
            }
            predictionVectorizer.setPriors = new HashMap<>();
            predictionVectorizer.cardPriors = new HashMap<>();
            for (MultiLabel multiLabel : dataSet.getMultiLabels()) {
                double setCount = predictionVectorizer.setPriors.getOrDefault(multiLabel, 0.0);
                predictionVectorizer.setPriors.put(multiLabel, setCount + 1.0 / dataSet.getNumDataPoints());
                double cardCount = predictionVectorizer.cardPriors.getOrDefault(multiLabel.getNumMatchedLabels(), 0.0);
                predictionVectorizer.cardPriors.put(multiLabel.getNumMatchedLabels(), cardCount + 1.0 / dataSet.getNumDataPoints());
            }
            predictionVectorizer.labelCalibrator = labelCalibrator;

            return predictionVectorizer;
        }


        public PredictionVectorizer build(MultiLabelClfDataSet dataSet, LabelCalibrator labelCalibrator, Hierarchy hierarchy) {
            PredictionVectorizer predictionVectorizer = new PredictionVectorizer(this);
            if (implication) {
                predictionVectorizer.implications = findImplications(dataSet.getMultiLabels(), dataSet.getNumClasses());
            }

            if (pairPrior) {
                predictionVectorizer.pairPriors = computePairPriors(dataSet.getMultiLabels(), dataSet.getNumClasses());
            }
            predictionVectorizer.setPriors = new HashMap<>();
            predictionVectorizer.cardPriors = new HashMap<>();
            for (MultiLabel multiLabel : dataSet.getMultiLabels()) {
                double setCount = predictionVectorizer.setPriors.getOrDefault(multiLabel, 0.0);
                predictionVectorizer.setPriors.put(multiLabel, setCount + 1.0 / dataSet.getNumDataPoints());
                double cardCount = predictionVectorizer.cardPriors.getOrDefault(multiLabel.getNumMatchedLabels(), 0.0);
                predictionVectorizer.cardPriors.put(multiLabel.getNumMatchedLabels(), cardCount + 1.0 / dataSet.getNumDataPoints());
            }
            predictionVectorizer.labelCalibrator = labelCalibrator;
            predictionVectorizer.hierarchyRelation = hierarchy;
            return predictionVectorizer;
        }
    }
}
