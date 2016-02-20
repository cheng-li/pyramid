package edu.neu.ccs.pyramid.experiment;

import org.apache.commons.io.FileUtils;

import java.io.File;

/**
 * generate exp211 configs for scene
 * Created by chengli on 2/1/16.
 */
public class Exp153 {
    public static void main(String[] args) throws Exception{
        int[] numClusters = {1,5,10,15,20,25,30,35,40,45,50,55,60};
        int configStart = 71;
        String out = "/Users/chengli/tmp/scene";

        int[] configs = new int[numClusters.length];
        for (int i=0;i<numClusters.length;i++){
            configs[i] = configStart+i;
        }

        for (int i=0;i<numClusters.length;i++){
            int numCluster = numClusters[i];
            int configIndex = configs[i];
            String configString = generate(numCluster,configIndex);
            File file = new File(out,""+configIndex);
            FileUtils.writeStringToFile(file,configString,false);
            File jobFile = new File(out,"job"+configIndex);
            String jobString = job(configIndex);
            FileUtils.writeStringToFile(jobFile,jobString,false);
        }

    }

    private static String generate(int numCluster, int configIndex){
        StringBuilder sb = new StringBuilder();
        sb.append("input.trainData=/scratch/wang.bin/trec.data/scene/data_sets/train").append("\n");
        sb.append("input.testData=/scratch/wang.bin/trec.data/scene/data_sets/test").append("\n");
        sb.append("input.matrixType=dense").append("\n");
        sb.append("output=/scratch/li.che/projects/pyramid/archives/exp211/scene/").append(configIndex).append("\n");
        sb.append("saveModel=true").append("\n");
        sb.append("saveModelForEachIter=true").append("\n");
        sb.append("modelName=model").append("\n");
        sb.append("mixture.numClusters=").append(numCluster).append("\n");
        sb.append("mixture.multiClassClassifierType=lr").append("\n");
        sb.append("mixture.binaryClassifierType=lr").append("\n");
        sb.append("train.warmStart=auto").append("\n");
        sb.append("train.initialize=true").append("\n");
        sb.append("predict.mode=dynamic").append("\n");
        sb.append("predict.sampling.numSamples=100").append("\n");
        sb.append("predict.allowEmpty=auto").append("\n");
        sb.append("em.startTemperature=1.0").append("\n");
        sb.append("em.endTemperature=1.0").append("\n");
        sb.append("em.numTemperatures=1").append("\n");
        sb.append("em.numIterations=50").append("\n");
        sb.append("lr.multiClassVariance=1").append("\n");
        sb.append("lr.binaryVariance=1").append("\n");
        sb.append("lr.meanRegularization=false").append("\n");
        sb.append("lr.meanRegVariance=1.0").append("\n");
        sb.append("boost.numIterationsBinary=20").append("\n");
        sb.append("boost.numIterationsMultiClass=20").append("\n");
        sb.append("boost.numLeavesBinary=2").append("\n");
        sb.append("boost.numLeavesMultiClass=2").append("\n");
        sb.append("boost.shrinkageBinary=0.1").append("\n");
        sb.append("boost.shrinkageMultiClass=0.1").append("\n");
        return sb.toString();
    }

    private static String job(int configIndex){
        StringBuilder sb = new StringBuilder();
        sb.append("#!/bin/sh").append("\n");
        sb.append("#BSUB -q ser-par-10g-4").append("\n");
        sb.append("#BSUB -n 48").append("\n");
        sb.append("#BSUB -R span[ptile=48]").append("\n");
        sb.append("#BSUB -cwd /scratch/li.che/pyramid/exp211").append("\n");
        sb.append("#BSUB -oo /scratch/li.che/pyramid/exp211/logs/scene/").append(configIndex).append(".log").append("\n");
        sb.append("./run.sh configs/scene/").append(configIndex).append(" >> logs/scene/").append(configIndex).append("\n");
        return sb.toString();
    }
}
