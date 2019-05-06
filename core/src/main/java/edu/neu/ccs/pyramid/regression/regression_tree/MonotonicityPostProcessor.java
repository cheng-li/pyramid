package edu.neu.ccs.pyramid.regression.regression_tree;

import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.PrintUtil;
import org.ojalgo.matrix.store.PhysicalStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.optimisation.Optimisation;
import org.ojalgo.optimisation.convex.ConvexSolver;

import java.util.Arrays;
import java.util.List;

public class MonotonicityPostProcessor {

    /**
     * use quadratic programming to find leaf nodes outputs
     * @param leaves
     * @param monotonicity
     */
    public static void changeOutput(List<Node> leaves, int[] monotonicity){
        List<Pair<Integer,Integer>> constraints = MonotonicityConstraintFinder.findComparablePairs(leaves, monotonicity);
        if (!MonotonicityConstraintFinder.isMonotonic(leaves,monotonicity)){
            System.out.println("NOT monotonic");
        }
        System.out.println("constraints = "+constraints);

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
        ConvexSolver convexSolver = ConvexSolver.getBuilder()
                .objective(Q,C)
                .inequalities(A,b)
                .build();

        Optimisation.Result result = convexSolver.solve();

        for (int i=0;i<leaves.size();i++){
            Node leaf = leaves.get(i);
            leaf.setValue(result.doubleValue(i));
        }

        System.out.println("after fixing");
        System.out.println(PrintUtil.printWithIndex(leaves.stream().mapToDouble(l->l.getValue()).toArray()));
    }
}
