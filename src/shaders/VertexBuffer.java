package shaders;

import static org.lwjgl.opengl.GL15.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15.glBindBuffer;
import static org.lwjgl.opengl.GL15.glBufferData;
import static org.lwjgl.opengl.GL30.*;

public class VertexBuffer {
	private int vao;
	private int vbo;
	
	public VertexBuffer() { 
		vao = glGenVertexArrays();
		glBindVertexArray(vao);
		vbo = glGenBuffers();
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	
	public void setData(float[] data) {
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW);
	}
	public void setSubData(int index, float[] subdata) {
		glBufferSubData(GL_ARRAY_BUFFER, index, subdata);
	}
	
}
