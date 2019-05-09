package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.PrintUtil;
import org.ojalgo.matrix.store.PhysicalStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.optimisation.Optimisation;
import org.ojalgo.optimisation.convex.ConvexSolver;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

public class MonotonicityPostProcessor implements Callable<List<Double>> {
    private List<Node> leaves;
    private int[] monotonicity;

    public MonotonicityPostProcessor(List<Node> leaves, int[] monotonicity) {
        this.leaves = leaves;
        this.monotonicity = monotonicity;
    }

    //
//    /**
//     * use quadratic programming to find leaf nodes outputs
//     * @param leaves
//     * @param monotonicity
//     */
//    public static void changeOutput(List<Node> leaves, int[] monotonicity){
//        List<Pair<Integer,Integer>> constraints = MonotonicityConstraintFinder.findComparablePairs(leaves, monotonicity);
//
//        //todo debugging
//        List<Pair<Integer,Integer>> violatingPairs = MonotonicityConstraintFinder.findViolatingPairs(leaves, monotonicity);
//        if (!violatingPairs.isEmpty()){
//            System.out.println("NOT monotonic");
//        }
//        System.out.println("constraints = "+constraints);
//        for (Pair<Integer,Integer> pair: constraints){
//            System.out.println("-----------------");
//            Bounds bound1 = new Bounds(leaves.get(pair.getFirst()));
//            Bounds bound2 = new Bounds(leaves.get(pair.getSecond()));
//            System.out.println("smaller node "+pair.getFirst()+" = "+bound1);
//            System.out.println("bigger node "+pair.getSecond()+" = "+bound2);
//            System.out.println("-----------------");
//        }
//
//
//        System.out.println("violating pairs = "+violatingPairs);
//
//        PhysicalStore.Factory<Double, PrimitiveDenseStore> storeFactory = PrimitiveDenseStore.FACTORY;
//        int numNodes = leaves.size();
//        double[] counts = new double[numNodes];
//        for (int i=0;i<leaves.size();i++){
//            counts[i] = MathUtil.arraySum(leaves.get(i).getProbs());
//        }
//
//        double[] means = new double[numNodes];
//        for (int i=0;i<leaves.size();i++){
//            means[i] = leaves.get(i).getValue();
//        }
//        System.out.println("before fixing");
//        System.out.println(PrintUtil.printWithIndex(means));
//
//        PrimitiveDenseStore Q = storeFactory.makeZero(numNodes,numNodes);
//        for (int i=0;i<numNodes;i++){
//            Q.add(i,i,counts[i]);
//        }
//
//        PrimitiveDenseStore C = storeFactory.makeZero(numNodes,1);
//        for (int i=0;i<numNodes;i++){
//            C.add(i,0,counts[i]*means[i]);
//        }
//
//        PrimitiveDenseStore A = storeFactory.makeZero(constraints.size(),numNodes);
//        for (int c=0;c<constraints.size();c++){
//            Pair<Integer,Integer> pair = constraints.get(c);
//            int smaller = pair.getFirst();
//            int bigger = pair.getSecond();
//            A.add(c,smaller,1);
//            A.add(c,bigger,-1);
//        }
//
//
//        PrimitiveDenseStore b = storeFactory.makeZero(constraints.size(),1);
//        Optimisation.Options options = new Optimisation.Options();
//        options.time_abort = 60000;
//        options.iterations_abort=1;
//
//        ConvexSolver convexSolver = ConvexSolver.getBuilder()
//                .objective(Q,C)
//                .inequalities(A,b)
//                .build(options);
//
//
//
//        Optimisation.Result result = convexSolver.solve();
//
//        for (int i=0;i<leaves.size();i++){
//            Node leaf = leaves.get(i);
//            leaf.setValue(result.doubleValue(i));
//        }
//
//        System.out.println("after fixing");
//        System.out.println(PrintUtil.printWithIndex(leaves.stream().mapToDouble(l->l.getValue()).toArray()));
//        if (MonotonicityConstraintFinder.isMonotonic(leaves, monotonicity)){
//            System.out.println("STILL NOT MONOTONIC!");
//        }
//    }

    @Override
    public List<Double> call() throws Exception {
        List<Pair<Integer,Integer>> constraints = MonotonicityConstraintFinder.findComparablePairs(leaves, monotonicity);

        //todo debugging
        List<Pair<Integer,Integer>> violatingPairs = MonotonicityConstraintFinder.findViolatingPairs(leaves, monotonicity);
        if (!violatingPairs.isEmpty()){
            System.out.println("NOT monotonic");
        }
        System.out.println("constraints = "+constraints);
        for (Pair<Integer,Integer> pair: constraints){
            System.out.println("-----------------");
            Bounds bound1 = new Bounds(leaves.get(pair.getFirst()));
            Bounds bound2 = new Bounds(leaves.get(pair.getSecond()));
            System.out.println("smaller node "+pair.getFirst()+" = "+bound1);
            System.out.println("bigger node "+pair.getSecond()+" = "+bound2);
            System.out.println("-----------------");
        }


        System.out.println("violating pairs = "+violatingPairs);

        PhysicalStore.Factory<Double, PrimitiveDenseStore> storeFactory = PrimitiveDenseStore.FACTORY;
        int numNodes = leaves.size();
        double[] counts = new double[numNodes];
        for (int i=0;i<leaves.size();i++){
            counts[i] = MathUtil.arraySum(leaves.get(i).getProbs());
        }

        double[] means = new double[numNodes];
        for (int i=0;i<leaves.size();i++){
            means[i] = leaves.get(i).getValue();
        }
        System.out.println("before fixing");
        System.out.println(PrintUtil.printWithIndex(means));

        PrimitiveDenseStore Q = storeFactory.makeZero(numNodes,numNodes);
        for (int i=0;i<numNodes;i++){
            Q.add(i,i,counts[i]);
        }

        PrimitiveDenseStore C = storeFactory.makeZero(numNodes,1);
        for (int i=0;i<numNodes;i++){
            C.add(i,0,counts[i]*means[i]);
        }

        PrimitiveDenseStore A = storeFactory.makeZero(constraints.size(),numNodes);
        for (int c=0;c<constraints.size();c++){
            Pair<Integer,Integer> pair = constraints.get(c);
            int smaller = pair.getFirst();
            int bigger = pair.getSecond();
            A.add(c,smaller,1);
            A.add(c,bigger,-1);
        }


        PrimitiveDenseStore b = storeFactory.makeZero(constraints.size(),1);
        Optimisation.Options options = new Optimisation.Options();
        options.time_abort = 60000;
        options.iterations_abort=1;

        ConvexSolver convexSolver = ConvexSolver.getBuilder()
                .objective(Q,C)
                .inequalities(A,b)
                .build(options);



        Optimisation.Result result = convexSolver.solve();

//        for (int i=0;i<leaves.size();i++){
//            Node leaf = leaves.get(i);
//            leaf.setValue(result.doubleValue(i));
//        }
        List<Double> list = new ArrayList<>();
        for (int i=0;i<leaves.size();i++){
            result.doubleValue(i);
        }

//        System.out.println("after fixing");
//        System.out.println(PrintUtil.printWithIndex(leaves.stream().mapToDouble(l->l.getValue()).toArray()));
//        if (MonotonicityConstraintFinder.isMonotonic(leaves, monotonicity)){
//            System.out.println("STILL NOT MONOTONIC!");
//        }
        return list;
    }
}
