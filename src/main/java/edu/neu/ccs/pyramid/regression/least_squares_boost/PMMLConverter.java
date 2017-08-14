package edu.neu.ccs.pyramid.regression.least_squares_boost;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.dmg.pmml.*;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.*;
import org.jpmml.converter.mining.MiningModelUtil;

import java.util.ArrayList;
import java.util.List;

/**
 * convert gb to pmml
 * based on jpmml
 */
public class PMMLConverter {

    public Label encodeLabel(FieldName targetField, List<String> targetCategories, PMMLEncoder encoder){
        if(targetCategories != null){
            throw new IllegalArgumentException();
        }

        DataField dataField = encoder.createDataField(targetField, OpType.CONTINUOUS, DataType.FLOAT);

        return new ContinuousLabel(dataField);
    }

    public MiningModel encodeMiningModel(List<RegressionTree> regTrees, float base_score, Schema schema){
        MiningModel miningModel = createMiningModel(regTrees, base_score, schema);

        return miningModel;
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

    public PMML encodePMML(FieldName targetField, List<String> targetCategories, FeatureList featureList, List<RegressionTree> regTrees, float base_score){
        LSBoostEncoder encoder = new LSBoostEncoder();

        if(targetField == null){
            targetField = FieldName.create("_target");
        }

        Label label = encodeLabel(targetField, targetCategories, encoder);

        //todo
        List<Feature> features = new ArrayList<>();


        Schema schema = new Schema(label, features);

        MiningModel miningModel = encodeMiningModel(regTrees, base_score, schema);

        PMML pmml = encoder.encodePMML(miningModel);

        return pmml;
    }


}
