package shaders;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glGetAttribLocation;
import static org.lwjgl.opengl.GL20.glGetUniformLocation;
import static org.lwjgl.opengl.GL20.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.*;

public class GLSLProgram {
	private int id;

	public GLSLProgram() {
		id = glCreateProgram();
	}
	
	public void use() {
		glLinkProgram(id);
		glUseProgram(id);
	}

	public void attachShader(int shader) {
		glAttachShader(id, shader);
	}

	public static int buildShaderFromPath(int shaderType, String path) {
		String data = null;
		try {
			data = new String(Files.readAllBytes(Paths.get(path)));
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		int shader = glCreateShader(shaderType);
		glShaderSource(shader, data);
		glCompileShader(shader);
		return shader;
	}
	public void vertexAttribLocation(String loc, int size, int type, boolean normalized, int stride, long pointer) {
		int ptr = glGetAttribLocation(id, loc);
		glEnableVertexAttribArray(ptr);
		glVertexAttribPointer(ptr, size, type, normalized, stride, pointer);
	}
	public int getUniformLocation(String param){
		return glGetUniformLocation(id, param);
	}
}
