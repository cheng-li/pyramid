package edu.neu.ccs.pyramid.multilabel_classification.powerset;

import edu.neu.ccs.pyramid.classification.lkboost.LKBOutputCalculator;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoost;
import edu.neu.ccs.pyramid.classification.lkboost.LKBoostOptimizer;
import edu.neu.ccs.pyramid.classification.logistic_regression.ElasticNetLogisticTrainer;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticLoss;
import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.ClfDataSet;
import edu.neu.ccs.pyramid.dataset.ClfDataSetBuilder;
import edu.neu.ccs.pyramid.dataset.MultiLabel;
import edu.neu.ccs.pyramid.dataset.MultiLabelClfDataSet;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.Overlap;
import edu.neu.ccs.pyramid.optimization.*;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import org.apache.mahout.math.Vector;

import java.awt.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Rainicy on 12/3/15.
 */
public class LPOptimizer {
    private LPClassifier classifier;

    private ClfDataSet dataSet;

    private MultiLabelClfDataSet testSet;

    private int numClasses;


    public LPOptimizer(LPClassifier classifier, MultiLabelClfDataSet dataSet, MultiLabelClfDataSet testSet) {
        this.classifier = classifier;
        this.testSet = testSet;
        // build index to multilabel and multilabel to index map.
        Map<Integer, MultiLabel> IDToML = new HashMap<>();
        Map<MultiLabel, Integer> MLToID = new HashMap<>();
        init(IDToML, MLToID, dataSet);

        this.classifier.setIDToML(IDToML);
        this.classifier.setMLToID(MLToID);

        System.out.println("number of new classes: " + IDToML.size());

        this.dataSet = ClfDataSetBuilder.getBuilder().numDataPoints(dataSet.getNumDataPoints())
                .numFeatures(dataSet.getNumFeatures()).dense(dataSet.isDense())
                .missingValue(dataSet.hasMissingValue()).numClasses(IDToML.size()).build();

        this.dataSet.setFeatureList(dataSet.getFeatureList());
        // todo buggy
        this.dataSet.setLabelTranslator(dataSet.getLabelTranslator());
        this.classifier.setFeatureList(this.dataSet.getFeatureList());
        //todo buggy
        this.classifier.setLabelTranslator(this.dataSet.getLabelTranslator());

        for (int n=0; n<dataSet.getNumDataPoints(); n++) {
            this.dataSet.setLabel(n, MLToID.get(dataSet.getMultiLabels()[n]));
            Vector feature = dataSet.getRow(n);
            for (Vector.Element element : feature.nonZeroes()) {
                this.dataSet.setFeatureValue(n, element.index(), element.get());
            }
        }

        this.numClasses = IDToML.size();

    }

    public void optimize(Config config) {
        String classifier = config.getString("classifier");

        if (classifier.equals("boost")) {
            LKBoost lkBoost = new LKBoost(numClasses);
            RegTreeConfig regTreeConfig = new RegTreeConfig()
                    .setMaxNumLeaves(config.getInt("numLeaves"));
            RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
            regTreeFactory.setLeafOutputCalculator(new LKBOutputCalculator(numClasses));

            LKBoostOptimizer optimizer = new LKBoostOptimizer(lkBoost, this.dataSet, regTreeFactory);
            optimizer.setShrinkage(0.1);
            optimizer.initialize();

            for (int round=0; round<config.getInt("numIters"); round++) {
                optimizer.iterate();
                this.classifier.estimator = lkBoost;
                System.out.print("round:"+round);
                System.out.print("\ttrainAcc: " + Accuracy.accuracy(lkBoost,dataSet));
                MultiLabel[] predictions = this.classifier.predict(testSet);
                System.out.print("\ttestAcc: " + Accuracy.accuracy(testSet.getMultiLabels(), predictions));
                System.out.println("\ttestOverlap: " + Overlap.overlap(testSet.getMultiLabels(), predictions));
            }
        } else if (classifier.equals("elasticnet")) {
            LogisticRegression logisticRegression = new LogisticRegression(numClasses, dataSet.getNumFeatures());
            ElasticNetLogisticTrainer optimizer = ElasticNetLogisticTrainer.newBuilder(logisticRegression, dataSet)
                    .setL1Ratio(config.getDouble("l1Ratio"))
                    .setRegularization(config.getDouble("regularization")).build();

            for (int round=0; round<config.getInt("numIters"); round++) {
                System.out.print("round:"+round);
                optimizer.iterate();
                this.classifier.estimator = logisticRegression;
                System.out.print("\ttrainAcc: " + Accuracy.accuracy(logisticRegression,dataSet));
                MultiLabel[] predictions = this.classifier.predict(testSet);
                System.out.print("\ttestAcc: " + Accuracy.accuracy(testSet.getMultiLabels(), predictions));
                System.out.println("\ttestOverlap: " + Overlap.overlap(testSet.getMultiLabels(), predictions));
            }
        } else if (classifier.equals("lr")){
            LogisticRegression logisticRegression = new LogisticRegression(numClasses, dataSet.getNumFeatures());
            LogisticLoss loss = new LogisticLoss(logisticRegression,dataSet, config.getDouble("variance"));
            LBFGS optimizer = new LBFGS(loss);
            optimizer.getTerminator().setAbsoluteEpsilon(0.1);
            for (int round=0; round<config.getInt("numIters"); round++) {
                optimizer.iterate();
                this.classifier.estimator = logisticRegression;
                System.out.print("round:"+round);
                System.out.print("\ttrainAcc: " + Accuracy.accuracy(logisticRegression,dataSet));
                MultiLabel[] predictions = this.classifier.predict(testSet);
                System.out.print("\ttestAcc: " + Accuracy.accuracy(testSet.getMultiLabels(), predictions));
                System.out.println("\ttestOverlap: " + Overlap.overlap(testSet.getMultiLabels(), predictions));
                if (optimizer.getTerminator().shouldTerminate()){
                    break;
                }
            }
        } else {
            throw new RuntimeException("Unknown classifier");
        }
    }



    private void init(Map<Integer, MultiLabel> idToML, Map<MultiLabel, Integer> mlToID, MultiLabelClfDataSet dataSet) {
        init(idToML, mlToID, dataSet.getMultiLabels());
    }

    private void init(Map<Integer, MultiLabel> idToML, Map<MultiLabel, Integer> mlToID, MultiLabel[] labels) {
        int index = 0;
        for (MultiLabel label : labels) {
            if (mlToID.containsKey(label)) {
                continue;
            }
            idToML.put(index, label);
            mlToID.put(label, index);
            index++;
        }
    }
}
