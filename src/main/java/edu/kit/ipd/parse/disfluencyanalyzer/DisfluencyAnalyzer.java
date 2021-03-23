package edu.kit.ipd.parse.disfluencyanalyzer;

import java.util.ArrayList;
import java.util.List;

import org.kohsuke.MetaInfServices;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.kit.ipd.parse.disfluencyanalyzer.util.GraphUtils;
import edu.kit.ipd.parse.luna.agent.AbstractAgent;
import edu.kit.ipd.parse.luna.graph.INode;

// TODO: Replace Sysouts with logger.info or logg.debug or remove them

/**
 * This agent is responsible for detecting and correcting speech disfluencies.
 * The agent uses a classifier that marks each word with a tag. The tag
 * describes the belonging of a word to a disfluency type or to specific part of
 * the disfluency (with IOB-information). The agent also connects the correcting
 * part (reparans) of a disfluency with its corresponding erroneous part
 * (reparandum), if there is corresponding part. The class is the main class of
 * the package that is responsible for coordinating all processes.
 * 
 * @author Robert Hochweiss
 * @author Sebastian Weigelt
 *
 */
@MetaInfServices(AbstractAgent.class)
public class DisfluencyAnalyzer extends AbstractAgent {

	private static final Logger logger = LoggerFactory.getLogger(DisfluencyAnalyzer.class);
	private LSTMClassifier classifier;

	private boolean runBefore = false;

	@Override
	public void init() {
		setId("disfluencyAnalyzer");
		/*
		 * Load pretrained GloVe-model and the pretrained model for the classifier
		 */
		classifier = new LSTMClassifier();
	}

	@Override
	public void exec() {
		if (runBefore) {
			logger.info("Ran before, exiting!");
			return;
		}
		List<INode> wordList = new ArrayList<>();
		List<String> posList = new ArrayList<>();
		List<String> chunkIOBList = new ArrayList<>();
		List<String> dfList = new ArrayList<>();
		if (GraphUtils.extractInputFromGraph(getGraph(), wordList, posList, chunkIOBList)) {
			logger.info("Start guessing the disfluency tags for the input utterance...");
			List<String> words = new ArrayList<>();
			for (int i = 0; i < wordList.size(); i++) {
				words.add(wordList.get(i).getAttributeValue("value").toString());
			}
			dfList.addAll(classifier.tagInputSequenceRaw(words, posList, chunkIOBList));
			GraphUtils.printToGraph(graph, wordList, dfList);
			logger.info("All done !");
		} else {
			logger.info("No disfluency predictions necessary");
		}
		runBefore = true;
	}

}
