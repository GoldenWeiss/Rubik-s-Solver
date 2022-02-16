package cube;

import java.util.Random;

public enum RNotation {
	L, R, U, D, F, B,   // clockwise rotation
	Li, Ri,  Ui, Di, Fi, Bi; // counterclockwise rotation
	
	public static enum Axis { X, Y, Z }
	
	public static RNotation fromId(int id) {
		return values()[id];
	}
	public static int randomRotation() 
	{
		Random rand = new Random();
		return rand.nextInt(values().length);
	}
	public static RNotation invertedRotation(RNotation rotation) 
	{
		return RNotation.values()[(rotation.ordinal() + 6) % 12];
	}
	
}