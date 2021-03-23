package edu.kit.ipd.parse.disfluencyanalyzer.util;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// TODO: cleanup

/**
 * Utility class that contains methods to extract features out of words or to
 * compare words and their features like for example their parts of speech.
 * 
 * @author Robert Hochweiss
 * @author Sebastian Weigelt
 *
 */
public final class WordUtils {

	private static final Logger logger = LoggerFactory.getLogger(WordUtils.class);
	/**
	 * A list of all relevant filled pauses. It contains almost all filled pauses
	 * that occur in the Switchboard corpus. The other filled pauses are either very
	 * special cases or mistakes of the translators or they are also explicit
	 * editing terms or discours marker.
	 */
	public static List<String> FILLED_PAUSES = Arrays.asList("um", "uh", "eh", "uh-oh", "un", "er", "eh", "ehm", "em", "ah", "ahm", "oh",
			"uhm", "duh", "ooh", "gee", "hum", "hm", "ugh", "gosh", "huh", "ahm", "yah", "hmm");

	/**
	 * A list of words that are the most important part of their explicit editing
	 * phrases. They are also the most common.
	 */
	public static List<String> EXPLICIT_EDITING_TERMS = Arrays.asList("mean", "sorry", "excuse");

	private WordUtils() {
	}

	/**
	 * This methods compares the parts of speech (POS) of two words and checks
	 * whether their POS is similar like for example whether they are verbs.
	 * 
	 * @param leftPOS
	 *            the left to be considered part of speech
	 * @param rightPOS
	 *            the right to be considered part of speech
	 * @return true if the two parts of speech are similar otherwise false
	 */
	public static boolean hasSimilarPOS(String leftPOS, String rightPOS) {
		// if (!validPOS.contains(rightPOS) || !validPOS.contains(rightPOS)) {
		// String errorMSG = "One of the to be POS tags: " + leftPOS + " and " +
		// rightPOS + " is undefined";
		// logger.error(errorMSG);
		// throw new IllegalArgumentException(errorMSG);
		// }
		List<String> simPOSPrefixes = Arrays.asList("NN", "VB", "JJ", "RB", "PRP", "WP");
		// in special cases forbidden POS tags
		/*
		 * Verbs in present or past participle can not be considered for similarity if
		 * they are next to an other verb for the adjacent words case since they could
		 * be part of a combined verb
		 */
		List<String> forbiddenTags = Arrays.asList("VBG", "VBN");
		if (leftPOS.equals(rightPOS)) {
			return true;
		}
		if (forbiddenTags.contains(leftPOS) || forbiddenTags.contains(rightPOS)) {
			return false;
		}
		return (leftPOS.substring(0, 2).equals(rightPOS.substring(0, 2)) && simPOSPrefixes.contains(leftPOS.substring(0, 2)));
	}

	/**
	 * This method returns the number of the same character of 2 words.
	 * 
	 * @param left
	 *            the left to be compared string
	 * @param right
	 *            the right to be compared string
	 * @return the number of same chars between the left and the right string
	 */
	public static double getNumSameChars(String left, String right) {
		HashMap<Character, Integer> occurence = new HashMap<Character, Integer>();
		for (int i = 0; i < left.length(); i++) {
			char ch = left.charAt(i);
			if (!occurence.containsKey(ch)) {
				occurence.put(ch, 1);
			} else {
				occurence.put(ch, occurence.get(ch) + 1);
			}
		}
		int numSameChars = 0;
		for (int i = 0; i < right.length(); i++) {
			char ch = right.charAt(i);
			if (occurence.containsKey(ch)) {
				numSameChars++;
				if (occurence.get(ch) <= 1) {
					occurence.remove(ch);
				} else {
					occurence.put(ch, occurence.get(ch) - 1);
				}
			}
		}
		return numSameChars;
	}

}
