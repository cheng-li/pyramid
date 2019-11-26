//package edu.neu.ccs.pyramid.calibration;
//
//import edu.neu.ccs.pyramid.dataset.LabelTranslator;
//import edu.neu.ccs.pyramid.dataset.MultiLabel;
//import edu.neu.ccs.pyramid.feature.FeatureList;
//import edu.neu.ccs.pyramid.multilabel_classification.DynamicProgramming;
//import edu.neu.ccs.pyramid.multilabel_classification.MultiLabelClassifier;
//import edu.neu.ccs.pyramid.multilabel_classification.cbm.BMDistribution;
//import edu.neu.ccs.pyramid.util.Pair;
//import org.apache.mahout.math.Vector;
//
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Map;
//import java.util.Optional;
//
//public class RerankerGFM implements MultiLabelClassifier {
//    private Reranker reranker;
//
//    public RerankerGFM(Reranker reranker) {
//        this.reranker = reranker;
//    }
//
//    @Override
//    public int getNumClasses() {
//        return reranker.getNumClasses();
//    }
//
//    @Override
//    public MultiLabel predict(Vector vector) {
//        return reranker.predictByGFM(vector);
//    }
//
//
//    @Override
//    public FeatureList getFeatureList() {
//        return reranker.getFeatureList();
//    }
//
//    @Override
//    public LabelTranslator getLabelTranslator() {
//        return reranker.getLabelTranslator();
//    }
//}
