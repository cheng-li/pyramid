//package edu.neu.ccs.pyramid.multilabel_classification.bmm_variant;
//
///**
// * Created by Rainicy on 11/10/15.
// */
//public class BMMReports {// generate reports:
//    public void generateReports(MultiLabelClfDataSet dataSet, String reportsPath, double softmaxVariance, double logitVariance, Config config) throws IOException {
//
//
//        BMMOptimizer optimizer = new BMMOptimizer(this, dataSet, softmaxVariance, logitVariance);
//        optimizer.eStep();
//        double[][] gammas = optimizer.gammas;
//
//
//        File file = new File(reportsPath);
//        BufferedWriter bw = new BufferedWriter(new FileWriter(file));
//
//        bw.write("===========================CONFIG=============================\n");
//        bw.write(config.toString());
//        bw.write("===========================CONFIG=============================\n");
//        bw.write("\n\n\n");
//        for (int n=0; n<dataSet.getNumDataPoints(); n++) {
//            bw.write("data point: " + n + "\t" + "true label: " + dataSet.getMultiLabels()[n].toString() + "\n");
//
//            generateReportsForN(dataSet.getRow(n), dataSet.getMultiLabels()[n],gammas[n], bw, config.getInt("topM"));
//            bw.write("===============================================================\n");
//            bw.write("\n");
//            bw.write("===============================================================\n");
//        }
//
//        bw.close();
//    }
//
//    private void generateReportsForN(Vector vector, MultiLabel multiLabel, double[] gamma, BufferedWriter bw, int top) throws IOException {
//        double[] logisticProb = multiClassClassifier.predictClassProbs(vector);
//        bw.write("PIs: \t");
//        for (double piK : logisticProb) {
//            bw.write( String.format( "%.4f", piK) + "\t");
//        }
//        bw.write("\n");
//        bw.write("Gams: \t");
//        for (double gams : gamma) {
//            bw.write( String.format( "%.4f", gams) + "\t");
//        }
//        bw.write("\n");
//
//        // cache the prediction for binaryClassifiers[numClusters][numLabels]
//        double[][][] logProbsForX = new double[numClusters][numLabels][2];
//        for (int k=0; k<logProbsForX.length; k++) {
//            for (int l=0; l<logProbsForX[k].length; l++) {
//                logProbsForX[k][l] = binaryClassifiers[k][l].predictLogClassProbs(vector);
//            }
//        }
//
//        double[] logisticLogProb = multiClassClassifier.predictLogClassProbs(vector);
//        double topM;
//        if (top >= numClusters) {
//            topM = 0.0;
//        } else {
//            topM = getTopM(vector,logisticProb, top);
//        }
//
//        this.samplesForCluster = sampleFromSingles(vector, logisticProb, topM);
//
//        Map<MultiLabel, Double> mapMixValue = new HashMap<>();
//        Map<MultiLabel, String> mapString = new HashMap<>();
//        for (MultiLabel label : this.samplesForCluster) {
//            Vector candidateY = new DenseVector(numLabels);
//            for(int labelIndex : label.getMatchedLabels()) {
//                candidateY.set(labelIndex, 1.0);
//            }
//
//            double[] logPYnk = clusterConditionalLogProbArr(logProbsForX,candidateY);
//            double[] sumLog = new double[logisticLogProb.length];
//            for (int k=0; k<numClusters; k++) {
//                sumLog[k] = logisticLogProb[k] + logPYnk[k];
//            }
//            double logProb = MathUtil.logSumExp(sumLog);
//
//            String eachLine = label.toString() + "\t";
//            for (int k=0; k<numClusters; k++) {
//                eachLine +=  String.format( "%.4f", Math.exp(logPYnk[k])) + "\t";
//            }
//            eachLine +=  String.format( "%.4f", Math.exp(logProb)) + "\n";
//
//            mapString.put(label, eachLine);
//            mapMixValue.put(label, logProb);
//        }
//        MyComparator comp=new MyComparator(mapMixValue);
//        Map<MultiLabel,Double> sortedMap = new TreeMap(comp);
//        sortedMap.putAll(mapMixValue);
//
//        for (Map.Entry<MultiLabel, Double> entry : sortedMap.entrySet()) {
//            bw.write(mapString.get(entry.getKey()));
//        }
//
//        bw.write("------------------------------------\n");
//        Set<MultiLabel> trueSamples = new LinkedHashSet<>();
//        trueSamples.add(multiLabel);
//        for (int l : multiLabel.getMatchedLabels()) {
//            MultiLabel label = new MultiLabel();
//            label.addLabel(l);
//            trueSamples.add(label);
//        }
//        for (MultiLabel label : trueSamples) {
//            Vector candidateY = new DenseVector(numLabels);
//            for(int labelIndex : label.getMatchedLabels()) {
//                candidateY.set(labelIndex, 1.0);
//            }
//            double[] logPYnk = clusterConditionalLogProbArr(logProbsForX,candidateY);
//            double[] sumLog = new double[logisticLogProb.length];
//            for (int k=0; k<numClusters; k++) {
//                sumLog[k] = logisticLogProb[k] + logPYnk[k];
//            }
//            double logProb = MathUtil.logSumExp(sumLog);
//            bw.write(label.toString() + "\t");
//            for (int k=0; k<numClusters; k++) {
//                bw.write(String.format( "%.4f", Math.exp(logPYnk[k])) + "\t");
//            }
//            bw.write(String.format( "%.4f", Math.exp(logProb)) + "\n");
//        }
//    }
//
//    private double getTopM(Vector vector, double[] logisticProb, int m) throws IOException {
//        Map<Integer, Double> map = new HashMap<>();
//        for (int k=0; k<numClusters; k++) {
//            double maxProb = 1.0;
//            for (int l=0; l<numLabels; l++) {
//                double prob = binaryClassifiers[k][l].predictClassProbs(vector)[1];
//                if (prob > 0.5) {
//                    maxProb *= prob;
//                } else {
//                    maxProb *= (1-prob);
//                }
//            }
//            maxProb *= logisticProb[k];
//            map.put(k, maxProb);
//        }
//
//        MyComparator comp=new MyComparator(map);
//        Map<Integer,Double> sortedMap = new TreeMap(comp);
//        sortedMap.putAll(map);
//
//        int count=1;
//        double result = 0.0;
//        for (Map.Entry<Integer, Double> entry : sortedMap.entrySet()) {
//            if (count == m) {
//                result = entry.getValue();
//            }
//            count++;
//        }
//
//        return result;
//    }
//
//
//
//    static class MyComparator implements Comparator {
//
//        Map map;
//
//        public MyComparator(Map map) {
//            this.map = map;
//        }
//
//        public int compare(Object o1, Object o2) {
//
//            return ((Double) map.get(o2)).compareTo((Double) map.get(o1));
//
//        }
//    }
//}
