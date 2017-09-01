package edu.neu.ccs.pyramid.classification.lkboost;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import org.dmg.pmml.MathContext;
import org.dmg.pmml.MiningFunction;
import org.dmg.pmml.Model;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.True;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.regression.RegressionModel;
import org.dmg.pmml.regression.RegressionTable;
import org.jpmml.converter.CategoricalLabel;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.ModelUtil;
import org.jpmml.converter.Schema;
import org.jpmml.converter.ValueUtil;
import org.jpmml.converter.regression.RegressionModelUtil;

/**
 * This is copied from org.jpmml.converter.mining.MiningModelUtil
 * We modify the source code to make softmax work with binary classification
 */
public class MiningModelUtil {

    private MiningModelUtil(){
    }



    static
    public MiningModel createClassification(List<? extends Model> models, RegressionModel.NormalizationMethod normalizationMethod, boolean hasProbabilityDistribution, Schema schema){
        CategoricalLabel categoricalLabel = (CategoricalLabel)schema.getLabel();

        // modified here
        if(categoricalLabel.size() != models.size()){
            throw new IllegalArgumentException();
        } // End if

        if(normalizationMethod != null){

            switch(normalizationMethod){
                case NONE:
                case SIMPLEMAX:
                case SOFTMAX:
                    break;
                default:
                    throw new IllegalArgumentException();
            }
        }

        MathContext mathContext = null;

        List<RegressionTable> regressionTables = new ArrayList<>();

        for(int i = 0; i < categoricalLabel.size(); i++){
            Model model = models.get(i);

            MathContext modelMathContext = model.getMathContext();
            if(modelMathContext == null){
                modelMathContext = MathContext.DOUBLE;
            } // End if

            if(mathContext == null){
                mathContext = modelMathContext;
            } else

            {
                if(!Objects.equals(mathContext, modelMathContext)){
                    throw new IllegalArgumentException();
                }
            }

            Feature feature = MODEL_PREDICTION.apply(model);

            RegressionTable regressionTable = RegressionModelUtil.createRegressionTable(Collections.singletonList(feature), Collections.singletonList(1d), null)
                    .setTargetCategory(categoricalLabel.getValue(i));

            regressionTables.add(regressionTable);
        }

        RegressionModel regressionModel = new RegressionModel(MiningFunction.CLASSIFICATION, ModelUtil.createMiningSchema(categoricalLabel), regressionTables)
                .setNormalizationMethod(normalizationMethod)
                .setMathContext(ModelUtil.simplifyMathContext(mathContext))
                .setOutput(hasProbabilityDistribution ? ModelUtil.createProbabilityOutput(mathContext, categoricalLabel) : null);

        List<Model> segmentationModels = new ArrayList<>(models);
        segmentationModels.add(regressionModel);

        return createModelChain(segmentationModels, schema);
    }

    static
    public MiningModel createModelChain(List<? extends Model> models, Schema schema){

        if(models.size() < 1){
            throw new IllegalArgumentException();
        }

        Segmentation segmentation = createSegmentation(Segmentation.MultipleModelMethod.MODEL_CHAIN, models);

        Model lastModel = Iterables.getLast(models);

        MiningModel miningModel = new MiningModel(lastModel.getMiningFunction(), ModelUtil.createMiningSchema(schema.getLabel()))
                .setMathContext(ModelUtil.simplifyMathContext(lastModel.getMathContext()))
                .setSegmentation(segmentation);

        return miningModel;
    }

    static
    public Segmentation createSegmentation(Segmentation.MultipleModelMethod multipleModelMethod, List<? extends Model> models){
        return createSegmentation(multipleModelMethod, models, null);
    }

    static
    public Segmentation createSegmentation(Segmentation.MultipleModelMethod multipleModelMethod, List<? extends Model> models, List<? extends Number> weights){

        if((weights != null) && (models.size() != weights.size())){
            throw new IllegalArgumentException();
        }

        List<Segment> segments = new ArrayList<>();

        for(int i = 0; i < models.size(); i++){
            Model model = models.get(i);
            Number weight = (weights != null ? weights.get(i) : null);

            Segment segment = new Segment()
                    .setId(String.valueOf(i + 1))
                    .setPredicate(new True())
                    .setModel(model);

            if(weight != null && !ValueUtil.isOne(weight)){
                segment.setWeight(ValueUtil.asDouble(weight));
            }

            segments.add(segment);
        }

        return new Segmentation(multipleModelMethod, segments);
    }

    private static final Function<Model, Feature> MODEL_PREDICTION = new Function<Model, Feature>(){

        @Override
        public Feature apply(Model model){
            Output output = model.getOutput();

            if(output == null || !output.hasOutputFields()){
                throw new IllegalArgumentException();
            }

            OutputField outputField = Iterables.getLast(output.getOutputFields());

            return new ContinuousFeature(null, outputField.getName(), outputField.getDataType());
        }
    };
}
