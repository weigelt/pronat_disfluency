package edu.kit.ipd.parse.disfluencyanalyzer.util;

import java.util.List;

import edu.kit.ipd.parse.luna.graph.IArc;
import edu.kit.ipd.parse.luna.graph.IArcType;
import edu.kit.ipd.parse.luna.graph.IGraph;
import edu.kit.ipd.parse.luna.graph.INode;
import edu.kit.ipd.parse.luna.graph.INodeType;

/*
 * TODO: More code cleanup, 
 * change accessing process for the graph (with next arc between the nodes), 
 * add more conditions for checking  whether the graph has changed, 
 * maybe a different search process for corresponding RS-RM blocks ?
 */

/**
 * A utility class that provides some methods for accessing and modifying the
 * PARSE graph.
 * 
 * @author Robert Hochweiss
 * @author Sebastian Weigelt
 *
 */
public final class GraphUtils {

	private GraphUtils() {
	}

	/**
	 * This method extracts the information from the PARSE graph that are required
	 * for the disfluency classification of an utterance. It extracts for each word
	 * of an utterance the word itself, its part of speech (POS) tag and its chunk
	 * IOB tag. It saves this information in the given lists. It returns true if the
	 * classifier has to predict the disfluency tags for the utterance (new
	 * utterance or a change in the graph) otherwise false.
	 * 
	 * @param graph
	 *            the graph that contains the input information that is to be
	 *            extracted
	 * @param tokenList
	 *            the list, where the extracted input word of an utterance are to be
	 *            saved
	 * @param posList
	 *            the list, where the extracted input part of speech tags are to be
	 *            saved
	 * @param chunkIOBList
	 *            the list, where the extracted input chunk IOB tags are to be saved
	 * @return true if the classifier has to predict the disfluency tags for the
	 *         utterance (new utterance or a change in the graph) otherwise false
	 */
	public static boolean extractInputFromGraph(IGraph graph, List<INode> tokenList, List<String> posList, List<String> chunkIOBList) {
		INodeType nodeType = graph.getNodeType("token");
		boolean hasToPredict = false;
		if (nodeType != null) {
			List<INode> tokenNodes = graph.getNodesOfType(nodeType);
			for (INode token : tokenNodes) {
				// Check if the tokens contains attribute for the disfluency
				// tags
				if (!token.getAttributeNames().contains("disfluencyTag")) {
					hasToPredict = true;
				}
			}
			if (hasToPredict) {
				for (INode token : tokenNodes) {
					tokenList.add(token);
					posList.add(token.getAttributeValue("pos").toString());
					chunkIOBList.add(token.getAttributeValue("chunkIOB").toString());
				}
			}

		}
		return hasToPredict;
	}

	// Implement it better later, graph can be changed !
	/**
	 * This method prints the result of the disfluency classification to the graph.
	 * It adds the attribute "disfluencyTag" to each token (it does not exist
	 * already). It also creates a "repair" arc that starts at the first word of an
	 * reparans an ends at the corresponding first word of an reprandum (if it does
	 * exist).
	 * 
	 * @param graph
	 *            the graph where the results are printed to
	 * @param tokenList
	 *            the list of tokens which were used for the disfluency
	 *            classification process
	 * @param predictions
	 *            the list of the predicted disfluency tags (as strings) for each
	 *            word
	 */
	public static void printToGraph(IGraph graph, List<INode> tokenList, List<String> predictions) {
		// Look for B-RS B-RM pairs
		INodeType nodeType = graph.getNodeType("token");
		if (nodeType != null) {
			if (!nodeType.containsAttribute("disfluencyTag", "String")) {
				nodeType.addAttributeToType("String", "disfluencyTag");
			}
			for (int i = 0; i < tokenList.size(); i++) {
				INode token = tokenList.get(i);
				token.setAttributeValue("disfluencyTag", predictions.get(i));
			}
			if (!graph.hasArcType("disfluency")) {
				IArcType dfArc = graph.createArcType("disfluency");
				dfArc.addAttributeToType("String", "value");
			}
			for (int g = tokenList.size() - 1; g > 0; g--) {
				if (tokenList.get(g).getAttributeValue("disfluencyTag") != null) {
					if (tokenList.get(g).getAttributeValue("disfluencyTag").toString().equals("B-RS")) {
						// Search (backwards) for the corresponding B-RM
						for (int t = g - 1; t >= 0; t--) {
							if (tokenList.get(t).getAttributeValue("disfluencyTag").toString().equals("B-RM")) {
								IArc repairArc = graph.createArc(tokenList.get(g), tokenList.get(t), graph.getArcType("disfluency"));
								repairArc.setAttributeValue("value", "REPAIR");
							}
							// Abort search if a prior reparans block is found
							// (no corresponding reparandum)
							if (tokenList.get(t).getAttributeValue("disfluencyTag").toString().equals("B-RS")
									|| tokenList.get(t).getAttributeValue("disfluencyTag").toString().equals("I-RS")) {
								break;
							}
						}
					}
				}
			}
		}
	}

}
