package edu.neu.ccs.pyramid.classification.lkboost;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.dmg.pmml.*;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.*;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.MetroJAXBUtil;

import javax.xml.bind.JAXBException;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * convert lk boosting to PMML
 * based on jpmml and jpmml-xgboost
 */
public class PMMLConverter {

    public static Label encodeLabel(FieldName targetField, List<String> targetCategories, PMMLEncoder encoder, int numClasses){
        targetCategories = prepareTargetCategories(targetCategories, numClasses);

        DataField dataField = encoder.createDataField(targetField, OpType.CATEGORICAL, DataType.STRING, targetCategories);

        return new CategoricalLabel(dataField);
    }

    private static List<String> prepareTargetCategories(List<String> targetCategories, int numClass){

        if(targetCategories != null){

            if(targetCategories.size() != numClass){
                throw new IllegalArgumentException();
            }

            return targetCategories;
        }

        List<String> result = new ArrayList<>();

        for(int i = 0; i < numClass; i++){
            result.add(String.valueOf(i));
        }

        return result;
    }

    public static MiningModel encodeMiningModel(List<List<RegressionTree>> regTrees, float base_score, Schema schema){
        Schema segmentSchema = new Schema(new ContinuousLabel(null, DataType.FLOAT), schema.getFeatures());

        List<MiningModel> miningModels = new ArrayList<>();

        CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();


        int numClasses = regTrees.size();
        for (int l=0;l<numClasses;l++){
            MiningModel miningModel = createMiningModel(regTrees.get(l), base_score, segmentSchema)
                    .setOutput(ModelUtil.createPredictedOutput(FieldName.create("class_(" + categoricalLabel.getValue(l) + ")"), OpType.CONTINUOUS, DataType.FLOAT));
            miningModels.add(miningModel);
        }

        return MiningModelUtil.createClassification(miningModels, RegressionModel.NormalizationMethod.SOFTMAX, true, schema);
    }

    static
    protected MiningModel createMiningModel(List<RegressionTree> regTrees, float base_score, Schema schema){
        ContinuousLabel continuousLabel = (ContinuousLabel)schema.getLabel();

        Schema segmentSchema = schema.toAnonymousSchema();

        List<TreeModel> treeModels = new ArrayList<>();

        for(RegressionTree regTree : regTrees){
            TreeModel treeModel = regTree.encodeTreeModel(segmentSchema);

            treeModels.add(treeModel);
        }

        MiningModel miningModel = new MiningModel(MiningFunction.REGRESSION, ModelUtil.createMiningSchema(continuousLabel))
                .setMathContext(MathContext.FLOAT)
                .setSegmentation(MiningModelUtil.createSegmentation(Segmentation.MultipleModelMethod.SUM, treeModels))
                .setTargets(ModelUtil.createRescaleTargets(null, ValueUtil.floatToDouble(base_score), continuousLabel));

        return miningModel;
    }

    public static PMML encodePMML(FieldName targetField, List<String> targetCategories, FeatureList featureList, List<List<RegressionTree>> regTrees, float base_score, int numClasses){
        LKBoostEncoder encoder = new LKBoostEncoder();

        if(targetField == null){
            targetField = FieldName.create("_target");
        }

        Label label = encodeLabel(targetField, targetCategories, encoder, numClasses);

        //todo
        List<Feature> features = new ArrayList<>();
        for (int i=0;i<featureList.size();i++){
            FieldName fieldName = new FieldName("feature_"+i);
            DataField dataField = encoder.createDataField(fieldName, OpType.CONTINUOUS, DataType.FLOAT);
            Feature feature = new ContinuousFeature(encoder, dataField);
            features.add(feature);
        }


        Schema schema = new Schema(label, features);

        MiningModel miningModel = encodeMiningModel(regTrees, base_score, schema);

        PMML pmml = encoder.encodePMML(miningModel);

        return pmml;
    }

    public static void savePMML(LKBoost lkBoost, File pmmlFile){
        FeatureList featureList = lkBoost.getFeatureList();
        List<List<RegressionTree>> regressionTrees = new ArrayList<>();
        for (int l=0;l<lkBoost.getNumClasses();l++){
            regressionTrees.add(new ArrayList<>());
            List<Regressor> regressors = lkBoost.getEnsemble(l).getRegressors();
            for (Regressor regressor: regressors){
                if (regressor instanceof ConstantRegressor){
                    RegressionTree constantTree = RegTreeTrainer.constantTree(((ConstantRegressor) regressor).getScore());
                    regressionTrees.get(l).add(constantTree);
                } else {
                    regressionTrees.get(l).add((RegressionTree) regressor);
                }
            }

        }


        PMML pmml = PMMLConverter.encodePMML(null, null, featureList, regressionTrees, 0, lkBoost.getNumClasses());

        try(OutputStream os = new FileOutputStream(pmmlFile)){
            MetroJAXBUtil.marshalPMML(pmml, os);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (JAXBException e) {
            e.printStackTrace();
        }
    }
}
