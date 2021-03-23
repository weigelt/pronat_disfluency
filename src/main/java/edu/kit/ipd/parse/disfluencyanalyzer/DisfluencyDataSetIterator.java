package edu.kit.ipd.parse.disfluencyanalyzer;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;

import edu.kit.ipd.parse.disfluencyanalyzer.util.WordUtils;
import info.debatty.java.stringsimilarity.JaroWinkler;

// TODO: Code cleanup, more functionility in other classes (discrete vector creation), sl4j logger instead of sysouts

/**
 * This class is responsible for providing the data set for the training and
 * evaluation process. It extracts the input features of an utterance (input
 * sequence). It also loads prepared features form the cleaned corpus files.
 * 
 * @author Robert Hochweiss
 * @author Sebastian Weigelt
 *
 */
public class DisfluencyDataSetIterator implements MultiDataSetIterator {

	private static final long serialVersionUID = 2243723028111900292L;
	private int batchSize;
	private int totalBatches;
	private int vectorSize;
	private int currentBatch;
	private int sequenceMaxLength;
	private int N_DISCRETE_FEATURES = 80;
	private List<List<String>> wordSequences; // Sequences of all input words,
												// one list per sequence.
												// Sequences of all input part of speech tags (input feature), one list per
												// sequence, one tag per word in the sequence.
	private List<List<Integer>> posSequences;
	// Sequences of all input Chunk-IOB tags (input feature), one list per
	// sequence, one tag per word in the sequence.
	private List<List<Integer>> chunkIOBSequences;
	// The corresponding DF tags for the word sequences, one list per sequence,
	// one tag per word in the sequence.
	private List<List<Integer>> dfSequences;
	private List<String> posTags; // All possible tags for POS
	private List<String> chunkIOBTags; // All possible tags for the Chunk-IOB
	private List<String> dfTags; // All possible tags for the DFs.
	private Map<String, INDArray> unknownWords;
	private WordVectors wordVectors;
	private MultiDataSetPreProcessor preProcessor;

	/**
	 * 
	 * @param batchSize
	 *            the number of sequences per trainings iteration (batch)
	 * @param sequenceMaxLength
	 *            max length in number of word for a sequence
	 * @param wordVectors
	 *            the lookup table for the word vectors
	 * @param posTags
	 *            all defined POS tags
	 * @param chunkIOBTags
	 *            all defined chunk IOB tags
	 * @param dfTags
	 *            all defined disfluency tags
	 */
	public DisfluencyDataSetIterator(int batchSize, int sequenceMaxLength, WordVectors wordVectors, List<String> posTags,
			List<String> chunkIOBTags, List<String> dfTags) {
		this.batchSize = batchSize;
		this.sequenceMaxLength = sequenceMaxLength;
		this.wordVectors = wordVectors;
		vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
		unknownWords = new HashMap<>(); // Needed because there is no good way
										// to extent an non Word2Vec vocab in
										// the framework.
		this.posTags = posTags;
		this.chunkIOBTags = chunkIOBTags;
		this.dfTags = dfTags;
		wordSequences = new ArrayList<>();
		posSequences = new ArrayList<>();
		chunkIOBSequences = new ArrayList<>();
		dfSequences = new ArrayList<>();
		totalBatches = (int) Math.ceil((double) wordSequences.size() / batchSize);
	}

	/**
	 * 
	 * @param wordList
	 * @param posList
	 * @param chunkIOBList
	 */
	public void addInputSequenceList(List<String> wordList, List<Integer> posList, List<Integer> chunkIOBList) {
		if (wordList != null) {
			for (int i = 0; i < wordList.size(); i++) {
				if (WordUtils.FILLED_PAUSES.contains(wordList.get(i))) {
					wordList.set(i, "uh");
				}
			}
			wordSequences.add(wordList);
		}
		if (posList != null) {
			posSequences.add(posList);
		}
		if (chunkIOBList != null) {
			chunkIOBSequences.add(chunkIOBList);
		}
		totalBatches = (int) Math.ceil((double) wordSequences.size() / batchSize);
	}

	@Override
	public boolean hasNext() {
		return (currentBatch < totalBatches);
	}

	@Override
	public MultiDataSet next() {
		return next(batchSize);
	}

	@Override
	public MultiDataSet next(int num) {
		int c = currentBatch * batchSize;
		int currentBatchSize = Math.min(batchSize, wordSequences.size() - c);
		List<List<String>> curWordSequences = new ArrayList<>();
		List<List<Integer>> curPOSSequences = new ArrayList<>();
		List<List<Integer>> curChunkIOBSequences = new ArrayList<>();
		List<List<Integer>> curDFSequences = new ArrayList<>();
		int maxLength = 0;
		// Get the word sequences and Dfs for the current batch
		for (int k = 0; k < currentBatchSize; k++) {
			curWordSequences.add(wordSequences.get(c + k));
			curPOSSequences.add(posSequences.get(c + k));
			curChunkIOBSequences.add(chunkIOBSequences.get(c + k));
			if (dfSequences.size() > 0) {
				curDFSequences.add(dfSequences.get(c + k));
			}
			maxLength = Math.max(maxLength, curWordSequences.get(k).size());
		}
		// INDArray for each input feature (word vectors, POS tags, Chunk-IOB
		// tags, the hand-crafted discrete features), the output (DF tags) and
		// the corresponding masks
		INDArray wordFeatures = Nd4j.create(new int[] { currentBatchSize, vectorSize, maxLength }, 'f');
		INDArray posFeatures = Nd4j.create(new int[] { currentBatchSize, posTags.size(), maxLength }, 'f');
		INDArray chunkIOBFeatures = Nd4j.create(new int[] { currentBatchSize, chunkIOBTags.size(), maxLength }, 'f');
		INDArray discreteFeatures = Nd4j.create(new int[] { currentBatchSize, N_DISCRETE_FEATURES, maxLength }, 'f');
		INDArray wordFeaturesMask = Nd4j.zeros(new int[] { currentBatchSize, maxLength }, 'f');
		INDArray posFeaturesMask = Nd4j.zeros(new int[] { currentBatchSize, maxLength }, 'f');
		INDArray chunkIOBFeaturesMask = Nd4j.zeros(new int[] { currentBatchSize, maxLength }, 'f');
		INDArray discreteFeaturesMask = Nd4j.zeros(new int[] { currentBatchSize, maxLength }, 'f');
		INDArray dfOutputTags = null;
		INDArray dfOutputTagsMask = null;
		if (curDFSequences.size() > 0) {
			dfOutputTags = Nd4j.create(new int[] { currentBatchSize, dfTags.size(), maxLength }, 'f');
			dfOutputTagsMask = Nd4j.zeros(new int[] { currentBatchSize, maxLength }, 'f');
		}

		// Iterate over all sequences of the current batch
		for (int i = 0; i < currentBatchSize; i++) {
			List<String> wordList = curWordSequences.get(i);
			List<Integer> posList = curPOSSequences.get(i);
			List<Integer> chunkIOBList = curChunkIOBSequences.get(i);
			List<Integer> dfList = null;
			/*
			 * The size is 0 if you want to predict an input sequence where there is no
			 * solution given
			 */
			if (curDFSequences.size() > 0) {
				dfList = curDFSequences.get(i);
			}
			// Iterate over all words (and other input features) of the current
			// sequence and modify the INDArrays accordingly
			for (int j = 0; j < wordList.size() && j < maxLength; j++) {
				INDArray wordVector = null;
				if (wordVectors.hasWord(wordList.get(j))) {
					wordVector = wordVectors.getWordVectorMatrix(wordList.get(j));
				} else {
					wordVector = unknownWords.get(wordList.get(j));
				}
				// Word vectors can be put directly in the INDArray at the
				// specific point
				wordFeatures.put(new INDArrayIndex[] { point(i), all(), point(j) }, wordVector);
				wordFeaturesMask.putScalar(new int[] { i, j }, 1.0);
				/*
				 * The POS and the ChunkIOB features and the output tags are one-hot encoded in
				 * the INDArray at the specific point
				 */
				posFeatures.putScalar(new int[] { i, posList.get(j), j }, 1.0);
				posFeaturesMask.putScalar(new int[] { i, j }, 1.0);
				chunkIOBFeatures.putScalar(new int[] { i, chunkIOBList.get(j), j }, 1.0);
				chunkIOBFeaturesMask.putScalar(new int[] { i, j }, 1.0);
				// Generate the discrete features vector
				INDArray discreteFeaturesVector = createDiscreteFeatureVector(j, wordList, posList, chunkIOBList);
				discreteFeatures.put(new INDArrayIndex[] { point(i), all(), point(j) }, discreteFeaturesVector);
				discreteFeaturesMask.putScalar(new int[] { i, j }, 1.0);
				if (dfList != null) {
					dfOutputTags.putScalar(new int[] { i, dfList.get(j), j }, 1.0);
					dfOutputTagsMask.putScalar(new int[] { i, j }, 1.0);
				}
			}
		}
		currentBatch++;
		return new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] { wordFeatures, posFeatures, chunkIOBFeatures, discreteFeatures },
				new INDArray[] { dfOutputTags },
				new INDArray[] { wordFeaturesMask, posFeaturesMask, chunkIOBFeaturesMask, discreteFeaturesMask },
				new INDArray[] { dfOutputTagsMask });
	}

	@Override
	public MultiDataSetPreProcessor getPreProcessor() {
		return preProcessor;
	}

	@Override
	public boolean resetSupported() {
		return true;
	}

	@Override
	public boolean asyncSupported() {
		return true;
	}

	@Override
	public void reset() {
		currentBatch = 0;
	}

	@Override
	public void setPreProcessor(MultiDataSetPreProcessor preprocessor) {
		preProcessor = preprocessor;
	}

	/*
	 * Method for generating an discrete input vector that contains the hand-crafted
	 * discrete features
	 */
	private INDArray createDiscreteFeatureVector(int curWordPos, List<String> wordList, List<Integer> posList, List<Integer> chunkIOBList) {
		double[][] featureBits = new double[1][N_DISCRETE_FEATURES];
		String curWord = wordList.get(curWordPos);
		int curPOS = posList.get(curWordPos);
		/*
		 * Check in the surroundings (15 words on each side) of the current word for
		 * duplicates regarding the word itself and its part-of-speech
		 */
		for (int i = 1; i <= 15; i++) {
			/*
			 * The first 30 feature bits are for word matches, the next 30 for
			 * part-of-speech matches
			 */
			if ((curWordPos - i) > 0) {
				// if (distance.similarity(curWord, wordList.get(curWordPos -
				// i)) > 0.8) {
				// featureBits[0][15 - i] = 1.0;
				// }
				if (curWord.equals(wordList.get(curWordPos - i))) {
					featureBits[0][15 - i] = 1.0;
				} else {
					String nextWord = wordList.get(curWordPos - i);
					int numSameChars = (int) WordUtils.getNumSameChars(curWord, nextWord);
					if (!(WordUtils.FILLED_PAUSES.contains(curWord) || WordUtils.FILLED_PAUSES.contains(nextWord))
							&& (curWord.startsWith(wordList.get(curWordPos - i)) || wordList.get(curWordPos - i).startsWith(curWord))
							&& (i < 3) && (curWord.length() > 1) && (wordList.get(curWordPos - i).length() > 1)) {
						featureBits[0][15 - i] = 1.0;
					}
				}
				if (curPOS == posList.get(curWordPos - i)) {
					featureBits[0][30 + 15 - i] = 1.0;
				}
			}
			if ((curWordPos + i) < wordList.size()) {
				// if (distance.similarity(curWord, wordList.get(curWordPos +
				// i)) > 0.8) {
				// featureBits[0][15 - 1 + i] = 1.0;
				// }
				if (curWord.equals(wordList.get(curWordPos + i))) {
					featureBits[0][15 - 1 + i] = 1.0;
				} else {
					String nextWord = wordList.get(curWordPos + i);
					//					int numSameChars = (int) WordUtils.getNumSameChars(curWord, nextWord);
					if (!(WordUtils.FILLED_PAUSES.contains(curWord) || WordUtils.FILLED_PAUSES.contains(nextWord))
							&& (wordList.get(curWordPos + i).startsWith(curWord) || curWord.startsWith(wordList.get(curWordPos + i)))
							&& (i < 3) && (curWord.length() > 1) && (wordList.get(curWordPos + i).length() > 1)) {
						featureBits[0][15 - 1 + i] = 1.0;
					}
				}
				if (curPOS == posList.get(curWordPos + i)) {
					featureBits[0][30 + 15 - 1 + i] = 1.0;
				}
			}
		}
		/*
		 * Now check in the surroundings (4 Words on each side) of the current word and
		 * the word after it for duplicates regarding both words and their
		 * part-of-speech (bigram-feature)
		 */
		if ((curWordPos + 1) < wordList.size()) {
			String nextWord = wordList.get(curWordPos + 1);
			int nextPOS = posList.get(curWordPos + 1);

			for (int i = 1; i <= 4; i++) {
				/*
				 * The first 8 feature bits are for bigram word matches, the next 8 for bigram
				 * part-of-speech matches
				 */
				if ((curWordPos - i) > 0) {
					// if ((distance.similarity(curWord, wordList.get(curWordPos
					// - i)) > 0.8)
					// && (distance.similarity(nextWord, wordList.get(curWordPos
					// + 1 - i)) > 0.8)) {
					// featureBits[0][60 + 4 - i] = 1.0;
					// }
					if ((curWord.equals(wordList.get(curWordPos - i)) && (nextWord.equals(wordList.get(curWordPos + 1 - i))))) {
						featureBits[0][60 + 4 - i] = 1.0;
					}
					if ((curPOS == posList.get(curWordPos - i)) && (nextPOS == posList.get(curWordPos + 1 - i))) {
						featureBits[0][60 + 8 + 4 - i] = 1.0;
					}
				}
				if ((curWordPos + 1 + i) < wordList.size()) {
					// if ((distance.similarity(curWord, wordList.get(curWordPos
					// + i)) > 0.8)
					// && (distance.similarity(nextWord, wordList.get(curWordPos
					// + 1 + i)) > 0.8)) {
					// featureBits[0][60 + 4 - 1 + i] = 1.0;
					// }
					if ((curWord.equals(wordList.get(curWordPos + i))) && (nextWord.equals(wordList.get(curWordPos + 1 + i)))) {
						featureBits[0][60 + 4 - 1 + i] = 1.0;
					}
					if ((curPOS == posList.get(curWordPos + i)) && (nextPOS == posList.get(curWordPos + 1 + i))) {
						featureBits[0][60 + 8 + 4 - 1 + i] = 1.0;
					}
				}
			}
		}
		/*
		 * The Jaro-Winkler-distance(/similarity) works better as similarity metric for
		 * strings then the original one from the paper (at least for PARSE sequences)
		 */
		JaroWinkler distance = new JaroWinkler();
		/*
		 * The last 4 feature bits are for checking the close surrounding (2 words on
		 * each side) of the current word for a similar surface string
		 */
		for (int i = 1; i <= 2; i++) {
			if ((curWordPos - i) > 0) {
				// double similarity = 2.0 * getNumSameChars(curWord,
				// wordList.get(curWordPos - i))
				// / (curWord.length() + wordList.get(curWordPos - i).length());
				double similarity = distance.similarity(curWord, wordList.get(curWordPos - i));
				if (similarity > 0.8) {
					featureBits[0][76 + 2 - i] = 1.0;
				}
			}
			if ((curWordPos + i) < wordList.size()) {
				// double similarity = 2.0 * getNumSameChars(curWord,
				// wordList.get(curWordPos + i))
				// / (curWord.length() + wordList.get(curWordPos + i).length());
				double similarity = distance.similarity(curWord, wordList.get(curWordPos + i));
				if (similarity > 0.8) {
					featureBits[0][76 + 2 - 1 + i] = 1.0;
				}
			}
		}
		INDArray vector = Nd4j.create(featureBits, 'f');
		return vector;
	}

}