package cube;

import shaders.*;

public class Sprite {
	private GLSLProgram program;
	public Sprite(GLSLProgram pProgram) {
		setProgram(pProgram);
	}
	public void setProgram(GLSLProgram pProgram) {
		program = pProgram;
	}

	public GLSLProgram getProgram() {
		return program;
	}

	public void update() {
	}

	public void draw() {
	}
}
