package edu.kit.ipd.parse.disfluencyanalyzer;

import edu.kit.ipd.pronat.graph_builder.GraphBuilder;
import edu.kit.ipd.pronat.prepipedatamodel.PrePipelineData;
import edu.kit.ipd.pronat.prepipedatamodel.tools.StringToHypothesis;
import edu.kit.ipd.pronat.shallow_nlp.ShallowNLP;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import edu.kit.ipd.parse.luna.data.MissingDataException;
import edu.kit.ipd.parse.luna.pipeline.PipelineStageException;

/**
 * Simple input/output test for the dfAnalyzery agent.
 * 
 * @author Robert Hochweiss
 * @author Sebastian Weigelt
 *
 */
public class SimpleNetworkPredictionTest {

	private DisfluencyAnalyzer dfAnalyzer;
	private ShallowNLP snlp;
	private GraphBuilder graphBuilder;

	@Before
	public void setUp() {
		dfAnalyzer = new DisfluencyAnalyzer();
		dfAnalyzer.init();
		snlp = new ShallowNLP();
		snlp.init();
		graphBuilder = new GraphBuilder();
		graphBuilder.init();
	}

	@Test
	public void ioTest() {
		PrePipelineData ppd2 = new PrePipelineData();
		String input2 = "put uh the green in the dishwasher";
		ppd2.setMainHypothesis(StringToHypothesis.stringToMainHypothesis(input2, false));
		try {
			snlp.exec(ppd2);
			graphBuilder.exec(ppd2);
		} catch (PipelineStageException e) {
			e.printStackTrace();
		}
		try {
			dfAnalyzer.setGraph(ppd2.getGraph());
		} catch (MissingDataException e) {
			e.printStackTrace();
		}
		dfAnalyzer.exec();
		Assert.assertEquals("FP",
				dfAnalyzer.getGraph().getNodesOfType(dfAnalyzer.getGraph().getNodeType("token")).get(1).getAttributeValue("disfluencyTag"));
		assert (true);
	}

}
