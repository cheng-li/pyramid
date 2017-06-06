package edu.neu.ccs.pyramid.esplugins;

import org.apache.lucene.index.*;
import org.apache.lucene.search.*;

import java.io.IOException;
import java.util.Objects;
import java.util.Set;

/**
 * A Query that matches documents containing a term. This may be combined with
 * other terms with a {@link BooleanQuery}.
 */
public class TermCountQuery extends Query {

    private final Term term;
    private final TermContext perReaderTermState;

    final class TermWeight extends Weight {
        private final TermContext termStates;
        private final boolean needsScores;

        public TermWeight(IndexSearcher searcher, boolean needsScores, TermContext termStates)
                throws IOException {
            super(TermCountQuery.this);
            if (needsScores && termStates == null) {
                throw new IllegalStateException("termStates are required when scores are needed");
            }
            this.needsScores = needsScores;
            this.termStates = termStates;
        }

        @Override
        public void extractTerms(Set<Term> terms) {
            terms.add(getTerm());
        }

        @Override
        public String toString() {
            return "weight(" + TermCountQuery.this + ")";
        }

        // not used
        @Override
        public float getValueForNormalization() {
            return 1f;
        }

        // not used
        @Override
        public void normalize(float queryNorm, float boost) {
            ;
        }

        @Override
        public Scorer scorer(LeafReaderContext context) throws IOException {
            assert termStates == null || termStates.wasBuiltFor(ReaderUtil.getTopLevelContext(context)) : "The top-reader used to create Weight is not the same as the current reader's top-reader (" + ReaderUtil.getTopLevelContext(context);;
            final TermsEnum termsEnum = getTermsEnum(context);
            if (termsEnum == null) {
                return null;
            }
            PostingsEnum docs = termsEnum.postings(null, needsScores ? PostingsEnum.FREQS : PostingsEnum.NONE);
            assert docs != null;
            return new TermCountScorer(this, docs);
        }

        /**
         * Returns a {@link TermsEnum} positioned at this weights Term or null if
         * the term does not exist in the given context
         */
        private TermsEnum getTermsEnum(LeafReaderContext context) throws IOException {
            if (termStates != null) {
                // TermQuery either used as a Query or the term states have been provided at construction time
                assert termStates.wasBuiltFor(ReaderUtil.getTopLevelContext(context)) : "The top-reader used to create Weight is not the same as the current reader's top-reader (" + ReaderUtil.getTopLevelContext(context);
                final TermState state = termStates.get(context.ord);
                if (state == null) { // term is not present in that reader
                    assert termNotInReader(context.reader(), term) : "no termstate found but term exists in reader term=" + term;
                    return null;
                }
                final TermsEnum termsEnum = context.reader().terms(term.field()).iterator();
                termsEnum.seekExact(term.bytes(), state);
                return termsEnum;
            } else {
                // TermQuery used as a filter, so the term states have not been built up front
                Terms terms = context.reader().terms(term.field());
                if (terms == null) {
                    return null;
                }
                final TermsEnum termsEnum = terms.iterator();
                if (termsEnum.seekExact(term.bytes())) {
                    return termsEnum;
                } else {
                    return null;
                }
            }
        }

        private boolean termNotInReader(LeafReader reader, Term term) throws IOException {
            // only called from assert
            // System.out.println("TQ.termNotInReader reader=" + reader + " term=" +
            // field + ":" + bytes.utf8ToString());
            return reader.docFreq(term) == 0;
        }

        @Override
        public Explanation explain(LeafReaderContext context, int doc) throws IOException {
            Scorer scorer = scorer(context);
            if (scorer != null) {
                int newDoc = scorer.iterator().advance(doc);
                if (newDoc == doc) {
                    return Explanation.match(scorer.freq(), "term frequency");
                }
            }
            return Explanation.noMatch("no matching term");
        }
    }

    /** Constructs a query for the term <code>t</code>. */
    public TermCountQuery(Term t) {
        term = Objects.requireNonNull(t);
        perReaderTermState = null;
    }

    /** Returns the term of this query. */
    public Term getTerm() {
        return term;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, boolean needsScores) throws IOException {
        final IndexReaderContext context = searcher.getTopReaderContext();
        final TermContext termState;
        if (perReaderTermState == null
                || perReaderTermState.wasBuiltFor(context) == false) {
            if (needsScores) {
                // make TermQuery single-pass if we don't have a PRTS or if the context
                // differs!
                termState = TermContext.build(context, term);
            } else {
                // do not compute the term state, this will help save seeks in the terms
                // dict on segments that have a cache entry for this query
                termState = null;
            }
        } else {
            // PRTS was pre-build for this IS
            termState = this.perReaderTermState;
        }

        return new TermCountQuery.TermWeight(searcher, needsScores, termState);
    }

    /** Prints a user-readable version of this query. */
    @Override
    public String toString(String field) {
        StringBuilder buffer = new StringBuilder();
        if (!term.field().equals(field)) {
            buffer.append(term.field());
            buffer.append(":");
        }
        buffer.append(term.text());
        return buffer.toString();
    }

    /** Returns true iff <code>o</code> is equal to this. */
    @Override
    public boolean equals(Object other) {
        return sameClassAs(other) &&
                term.equals(((TermCountQuery) other).term);
    }

    @Override
    public int hashCode() {
        return classHash() ^ term.hashCode();
    }
}
