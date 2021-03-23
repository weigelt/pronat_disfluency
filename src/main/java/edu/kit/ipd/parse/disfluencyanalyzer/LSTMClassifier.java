package edu.kit.ipd.parse.disfluencyanalyzer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.kit.ipd.pronat.prepipedatamodel.token.ChunkIOB;
import edu.kit.ipd.pronat.prepipedatamodel.token.POSTag;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.kit.ipd.parse.disfluencyanalyzer.util.WordUtils;
import org.nd4j.linalg.io.ClassPathResource;

/**
 * @author Robert Hochweiss
 * @author Sebastian Weigelt
 *
 */
public class LSTMClassifier {

	public static String wordVectorsPath = "";
	public static String modelSaveLocation = "";
	private List<String> posTags;
	private List<String> chunkIOBTags;
	private List<String> dfTags;
	private WordVectors wordVectors;
	private ComputationGraph net;
	private boolean fpToBeRemoved = true;

	public LSTMClassifier() {
		posTags = new ArrayList<>();
		// TODO: That's a bit hacky. The classifier expects 42 distinct PoS tags.
		//  However, the pre-pipeline models has 46, so we use the first 42.
		//  Don't know how to select the right ones.
		//  It works, don't worry!
		for (int i = 0; i < 42; i++) {
			posTags.add(POSTag.values()[i].getTag());
		}
		//		for (POSTag p : POSTag.values()) {
		//			posTags.add(p.getTag());
		//		}
		chunkIOBTags = new ArrayList<>();
		for (ChunkIOB c : ChunkIOB.values()) {
			chunkIOBTags.add(c.toString());
		}
		dfTags = new ArrayList<>();
		for (DisfluencyTag d : DisfluencyTag.values()) {
			dfTags.add(d.toString());
		}
		try {
			wordVectorsPath = new ClassPathResource("glove.twitter.27B.50d.txt").getFile().getAbsolutePath();
			modelSaveLocation = new ClassPathResource("DisfluencyModel.net").getFile().getAbsolutePath();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Loading word vectors.");
		wordVectors = WordVectorSerializer.loadStaticModel(new File(wordVectorsPath));
		System.out.println("Word vectors loaded.");
	}

	public List<Integer> tagInputSequence(List<String> wordList, List<Integer> posList, List<Integer> chunkIOBList) {
		/*
		 * Load the network from a saved model if there is no current network model
		 * configured and trained
		 */
		if (net == null) {
			try {
				net = ModelSerializer.restoreComputationGraph(modelSaveLocation);
				System.out.println("Saved model restored.");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		// the original positions of the FPs, first they have to be removed if
		// fpausesToBeRemoved is set then after the classification process
		// they have to be added again at their index
		List<Integer> fpIdxList = new ArrayList<>();
		if (fpToBeRemoved) {
			for (int i = 0; i < wordList.size(); i++) {
				if (WordUtils.FILLED_PAUSES.contains(wordList.get(i))) {
					fpIdxList.add(i);
					wordList.remove(i);
					posList.remove(i);
					chunkIOBList.remove(i);
				}
			}
		}
		List<Integer> predictionsList = new ArrayList<>();
		List<Integer> adjustedPredictions = new ArrayList<>();
		if (!wordList.isEmpty()) {
			// DisfluencyDataSetIterator for special case: only 1 sequence (for
			// usage, not for training/eval)
			DisfluencyDataSetIterator tagIterator = new DisfluencyDataSetIterator(1, 1000, wordVectors, posTags, chunkIOBTags, dfTags);
			tagIterator.addInputSequenceList(wordList, posList, chunkIOBList);
			INDArray networkOutput = net.outputSingle(tagIterator);
			// A row for each sequence, with 1 sequence only 1 row
			INDArray predictions = networkOutput.getRow(0);
			// 1 column per word of the sequence, the output prediction values
			// (probabilities) for the word are the rows
			for (int i = 0; i < predictions.columns(); i++) {
				INDArray guessCol = predictions.getColumn(i);
				// Get the index of the row (df output class) with the highest
				// probability
				int guessMaxIdx = Nd4j.getBlasWrapper().iamax(guessCol);
				predictionsList.add(guessMaxIdx);
			}
			// Adjust the predictions to remove some simple errors of the
			// predictions
			adjustedPredictions.addAll(adjustPredictions(wordList, posList, predictionsList));
		}
		if (!fpIdxList.isEmpty()) {
			for (int i = fpIdxList.size() - 1; i >= 0; i--) {
				adjustedPredictions.add(fpIdxList.get(i), dfTags.indexOf("FP"));
			}
		}
		return adjustedPredictions;
	}

	private List<Integer> adjustPredictions(List<String> wordList, List<Integer> posList, List<Integer> firstPredictions) {
		List<Integer> adjustedPredictions = new ArrayList<>();
		adjustedPredictions.addAll(firstPredictions);
		int idxBRM = dfTags.indexOf("B-RM");
		int idxIRM = dfTags.indexOf("I-RM");
		int idxBRS = dfTags.indexOf("B-RS");
		int idxIRS = dfTags.indexOf("I-RS");
		List<Integer> skipClasses = Arrays.asList(dfTags.indexOf("FP"), dfTags.indexOf("EE"), dfTags.indexOf("DM"));
		// Check for unrecognized repetitions
		for (int i = 0; i < wordList.size() - 1; i++) {
			if ((i + 1) < ((wordList.size() - 1))) {
				if (wordList.get(i).equals(wordList.get(i + 1)) || wordList.get(i + 1).startsWith(wordList.get(i))) {
					if (!(firstPredictions.get(i) == idxBRM)) {
						adjustedPredictions.set(i, idxBRM);
					}
					if (!(firstPredictions.get(i + 1) == idxBRS)) {
						adjustedPredictions.set(i + 1, idxBRS);
					}
				}
				if ((i + 2) < ((wordList.size() - 1))) {
					if ((skipClasses.contains(firstPredictions.get(i + 1)) && wordList.get(i).equals(wordList.get(i + 2)))
							|| (skipClasses.contains(firstPredictions.get(i + 1)) && wordList.get(i + 2).startsWith(wordList.get(i)))) {
						if (!(firstPredictions.get(i) == idxBRM)) {
							adjustedPredictions.set(i, idxBRM);
						}
						if (!(firstPredictions.get(i + 2) == idxBRS)) {
							adjustedPredictions.set(i + 2, idxBRS);
						}
					}
				}
			}
		}

		// Check for not correctly recognized explicit editing terms
		for (int i = 0; i < wordList.size(); i++) {
			if (WordUtils.EXPLICIT_EDITING_TERMS.contains(wordList.get(i))) {
				adjustedPredictions.set(i, dfTags.indexOf("EE"));
			}
		}

		// Now check for uncompleted RM or RS blocks
		for (int i = 0; i <= wordList.size() - 1; i++) {
			List<Integer> newTags = new ArrayList<>();
			boolean correspondenceReached = false;
			switch (dfTags.get(firstPredictions.get(i))) {
			case "B-RM":
				for (int j = i + 1; j < i + 3; j++) {
					if (j >= adjustedPredictions.size()) {
						break;
					}
					if (adjustedPredictions.get(j) == idxIRM) {
						continue;
					}
					if (adjustedPredictions.get(j) == idxBRS) {
						correspondenceReached = true;
						break;
					}
					if (adjustedPredictions.get(j) == idxBRM) {
						break;
					}
					if (wordList.get(i).equals(wordList.get(j))
							|| WordUtils.hasSimilarPOS(posTags.get(posList.get(i)), posTags.get(posList.get(j)))) {
						correspondenceReached = true;
						adjustedPredictions.set(j, idxBRS);
						break;
					}
					// add future probable I-RM tags
					if (!skipClasses.contains(adjustedPredictions.get(j))) {
						newTags.add(j);
					}
				}
				if (correspondenceReached) {
					for (int k : newTags) {
						adjustedPredictions.set(k, idxIRM);
					}
				}
				break;
			case "I-RM":
				if (i > 0) {
					// There cannot be a I-Tag without the corresponding B-Tag
					if (!((adjustedPredictions.get(i - 1) == idxBRM) || (adjustedPredictions.get(i - 1) == idxIRM))) {
						if (!skipClasses.contains(adjustedPredictions.get(i - 1))) {
							adjustedPredictions.set(i - 1, idxBRM);
						}

					}
				}
				// now search for the next B-RS
				int k = i + 1;
				boolean rsReached = false;
				for (int j = k; j < k + 5; j++) {
					if (j >= adjustedPredictions.size()) {
						break;
					}
					if (adjustedPredictions.get(j) == idxBRS) {
						rsReached = true;
						// save position of the B-RS tag
						k = j;
						break;
					}
				}
				if (rsReached) {
					for (int j = k + 1; j < k + 4; j++) {
						if (j >= adjustedPredictions.size()) {
							break;
						}
						if ((adjustedPredictions.get(j) == idxIRS) || (skipClasses.contains(adjustedPredictions.get(j)))) {
							continue;
						}
						if ((wordList.get(j).equals(wordList.get(i)))
								|| WordUtils.hasSimilarPOS(posTags.get(posList.get(j)), posTags.get(posList.get(i)))) {
							if (firstPredictions.get(j) == dfTags.indexOf("O-DF")) {
								adjustedPredictions.set(j, idxIRS);
							}
							break;
						}
					}
				}
				break;
			case "B-RS":
				for (int j = i - 1; j > i - 4; j--) {
					if (j < 0) {
						break;
					}
					if ((adjustedPredictions.get(j) == idxIRM)) {
						continue;
					}
					if (adjustedPredictions.get(j) == idxBRM) {
						correspondenceReached = true;
						break;
					}
					if ((adjustedPredictions.get(j) == idxBRS) || (adjustedPredictions.get(j) == idxIRS)) {
						break;
					}
					if (wordList.get(i).equals(wordList.get(j))
							|| WordUtils.hasSimilarPOS(posTags.get(posList.get(i)), posTags.get(posList.get(j)))) {
						correspondenceReached = true;
						adjustedPredictions.set(j, idxBRM);
						break;
					}
					// add future probable I-RM tags
					if (!skipClasses.contains(adjustedPredictions.get(j))) {
						newTags.add(j);
					}
				}
				if (correspondenceReached) {
					for (int m : newTags) {
						adjustedPredictions.set(m, idxIRM);
					}
				}
				break;
			// I-RS is too seldom correctly recognized to consider using it for
			// adjusting
			default:
				break;
			}
		}
		return adjustedPredictions;
	}

	/**
	 * This method first converts the raw String elements of the POS tag list and
	 * the list of Chunk tags into (categorical) indices. Then it tags the input
	 * sequence through the classifier with their disfluency tags. it returns the
	 * list of indices of the disfluency tags ordered according to their order
	 * occurrence in the input sequence.
	 * 
	 * @param wordList
	 * @param posList
	 * @param chunkIOBList
	 */
	public List<String> tagInputSequenceRaw(List<String> wordList, List<String> posList, List<String> chunkIOBList) {
		List<Integer> posIndexList = new ArrayList<>();
		List<Integer> chunkIOBIndexList = new ArrayList<>();
		for (String s : posList) {
			posIndexList.add(posTags.indexOf(s));
		}
		for (String s : chunkIOBList) {
			chunkIOBIndexList.add(chunkIOBTags.indexOf(s));
		}
		List<Integer> resultIndexes = tagInputSequence(wordList, posIndexList, chunkIOBIndexList);
		List<String> resultTags = new ArrayList<>();
		for (int i = 0; i < resultIndexes.size(); i++) {
			resultTags.add(dfTags.get(resultIndexes.get(i)));
		}
		return resultTags;
	}
}