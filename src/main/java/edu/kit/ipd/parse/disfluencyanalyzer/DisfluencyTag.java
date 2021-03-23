package edu.kit.ipd.parse.disfluencyanalyzer;

/**
 * This enumeration models the different states for a word regarding its
 * belonging to a disfluency type. The states indicate whether a word is the
 * beginning or inside of a reparandum, the beginning or inside a of reparans, a
 * filled pause, an explicit editing term, a discourse marker or if the word is
 * completely outside of a disfluence.
 * 
 * @author Robert Hochweiss
 * @author Sebastian Weigelt
 * 
 */
public enum DisfluencyTag {

	DISFLUENCY_OUTSIDE("O-DF"), REPARANDUM_BEGIN("B-RM"), REPARANDUM_INSIDE("I-RM"), REPARANS_BEGIN("B-RS"), REPARANS_INSIDE(
			"I-RS"), FILLED_PAUSE("FP"), EXPLICIT_EDITING_TERM("EE"), DISCOURSE_MARKER("DM");

	private final String tag;

	DisfluencyTag(String tag) {
		this.tag = tag;
	}

	/**
	 * 
	 * @param tag
	 * @return
	 */
	public static DisfluencyTag getTagObject(String tag) {
		switch (tag) {
		case "O-DF":
			return DISFLUENCY_OUTSIDE;
		case "B-RM":
			return REPARANDUM_BEGIN;
		case "I-RM":
			return REPARANDUM_INSIDE;
		case "B-RS":
			return REPARANS_BEGIN;
		case "I-RS":
			return REPARANS_INSIDE;
		case "FP":
			return FILLED_PAUSE;
		case "EE":
			return EXPLICIT_EDITING_TERM;
		case "DM":
			return DISCOURSE_MARKER;
		default:
			return null;
		}
	}

	/**
	 * 
	 * @return
	 */
	protected String getTag() {
		return tag;
	}

	@Override
	public String toString() {
		return getTag();
	}

}
