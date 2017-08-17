package edu.neu.ccs.pyramid.regression.least_squares_boost;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.regression.ConstantRegressor;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeTrainer;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.dmg.pmml.*;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.tree.TreeModel;
import org.jpmml.converter.*;
import org.jpmml.converter.mining.MiningModelUtil;
import org.jpmml.model.MetroJAXBUtil;

import javax.xml.bind.JAXBException;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static org.dmg.pmml.DataType.DOUBLE;
import static org.dmg.pmml.DataType.FLOAT;

/**
 * convert gb to pmml
 * based on jpmml
 */
public class PMMLConverter {

    public static Label encodeLabel(FieldName targetField, List<String> targetCategories, PMMLEncoder encoder){
        if(targetCategories != null){
            throw new IllegalArgumentException();
        }

        DataField dataField = encoder.createDataField(targetField, OpType.CONTINUOUS, DataType.FLOAT);

        return new ContinuousLabel(dataField);
    }

    public static MiningModel encodeMiningModel(List<RegressionTree> regTrees, float base_score, Schema schema){
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

    public static PMML encodePMML(FieldName targetField, List<String> targetCategories, FeatureList featureList, List<RegressionTree> regTrees, float base_score){
        LSBoostEncoder encoder = new LSBoostEncoder();

        if(targetField == null){
            targetField = FieldName.create("_target");
        }

        Label label = encodeLabel(targetField, targetCategories, encoder);

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

    public static void savePMML(LSBoost lsBoost, File pmmlFile){
        FeatureList featureList = lsBoost.getFeatureList();
        List<RegressionTree> regressionTrees = lsBoost.getEnsemble(0).getRegressors().stream()
                .filter(a->a instanceof RegressionTree).map(a->(RegressionTree)a).collect(Collectors.toList());

        double constant = ((ConstantRegressor)lsBoost.getEnsemble(0).get(0)).getScore();
        RegressionTree constantTree = RegTreeTrainer.constantTree(constant);
        List<RegressionTree> allTrees = new ArrayList<>();
        allTrees.add(constantTree);
        allTrees.addAll(regressionTrees);
        PMML pmml = PMMLConverter.encodePMML(null, null, featureList, allTrees, 0);


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
